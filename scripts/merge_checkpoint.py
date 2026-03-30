"""
Merge verl FSDP sharded checkpoints into HuggingFace format for vLLM inference.

Usage:
    python -m projects.finqa.scripts.merge_checkpoint \
        --ckpt_dir /path/to/checkpoints/finqa-grpo-incur/global_step_45/actor \
        --base_model_path /path/to/qwen \
        --output_dir /path/to/merged_model

The script:
1. Loads all model_world_size_N_rank_X.pt shard files
2. Merges ShardedTensor / flat tensor shards into full parameters
3. Saves as HuggingFace format (.safetensors) that vLLM can load
"""

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def inspect_shard(shard_path: str) -> dict:
    """Load a single shard file and report its format."""
    print(f"Inspecting shard: {shard_path}")
    state_dict = torch.load(shard_path, map_location="cpu", weights_only=False)

    print(f"  Number of keys: {len(state_dict)}")
    sample_keys = list(state_dict.keys())[:5]
    for key in sample_keys:
        val = state_dict[key]
        val_type = type(val).__name__
        if isinstance(val, torch.Tensor):
            print(f"  {key}: Tensor {val.shape} {val.dtype}")
        elif hasattr(val, "local_shards"):
            shards = val.local_shards()
            if shards:
                t = shards[0].tensor
                print(f"  {key}: ShardedTensor local_shard={t.shape} {t.dtype}, global_size={val.metadata().size}")
            else:
                print(f"  {key}: ShardedTensor (no local shards)")
        else:
            print(f"  {key}: {val_type}")

    return state_dict


def merge_sharded_tensors(all_shards: list[dict], key: str) -> torch.Tensor:
    """Merge ShardedTensor values from all ranks into a single full tensor."""
    # Collect all local shards across ranks
    pieces = []
    for rank, shard_dict in enumerate(all_shards):
        val = shard_dict[key]
        if hasattr(val, "local_shards"):
            local_shards = val.local_shards()
            for shard in local_shards:
                pieces.append({
                    "tensor": shard.tensor,
                    "offsets": shard.metadata.shard_offsets,
                    "sizes": shard.metadata.shard_sizes,
                })
            if rank == 0:
                global_size = val.metadata().size
        elif isinstance(val, torch.Tensor):
            pieces.append({"tensor": val, "rank": rank})
        else:
            raise ValueError(f"Unexpected type for key {key}: {type(val)}")

    if not pieces:
        raise ValueError(f"No shards found for key {key}")

    # If we have offset metadata (ShardedTensor), place shards at correct positions
    if "offsets" in pieces[0]:
        full_tensor = torch.zeros(global_size, dtype=pieces[0]["tensor"].dtype)
        for piece in pieces:
            offsets = piece["offsets"]
            sizes = piece["sizes"]
            tensor = piece["tensor"]

            if len(offsets) == 1:
                # 1D shard (flattened parameter)
                start = offsets[0]
                end = start + sizes[0]
                full_tensor[start:end] = tensor.flatten()[:sizes[0]]
            elif len(offsets) == 2:
                # 2D shard
                r_start, c_start = offsets
                r_end = r_start + sizes[0]
                c_end = c_start + sizes[1]
                full_tensor[r_start:r_end, c_start:c_end] = tensor.reshape(sizes)
            else:
                # Higher dimensional - use flat indexing
                flat_full = full_tensor.flatten()
                flat_offset = 0
                for d in range(len(offsets)):
                    flat_offset = flat_offset * global_size[d] + offsets[d]
                flat_size = 1
                for s in sizes:
                    flat_size *= s
                flat_full[flat_offset:flat_offset + flat_size] = tensor.flatten()[:flat_size]
                full_tensor = flat_full.reshape(global_size)

        return full_tensor
    else:
        # Plain tensors from each rank - concatenate and will reshape later
        return torch.cat([p["tensor"].flatten() for p in pieces], dim=0)


def merge_flat_tensors(all_shards: list[dict], key: str, target_shape: torch.Size) -> torch.Tensor:
    """Merge plain flat tensor shards by concatenating and reshaping."""
    tensors = [shard[key].flatten() for shard in all_shards]
    flat = torch.cat(tensors, dim=0)
    numel = 1
    for s in target_shape:
        numel *= s
    if flat.numel() == numel:
        return flat.reshape(target_shape)
    elif flat.numel() > numel:
        # Padding might have been added - take first numel elements
        return flat[:numel].reshape(target_shape)
    else:
        raise ValueError(
            f"Cannot reshape {key}: flat size {flat.numel()} vs target {numel} ({target_shape})"
        )


def merge_checkpoint(ckpt_dir: str, base_model_path: str, output_dir: str):
    """Merge FSDP sharded checkpoint into HuggingFace format."""
    ckpt_path = Path(ckpt_dir)
    output_path = Path(output_dir)

    # --- Find shard files ---
    shard_files = sorted(ckpt_path.glob("model_world_size_*_rank_*.pt"))
    if not shard_files:
        print(f"ERROR: No model shard files found in {ckpt_path}")
        print("Expected files like: model_world_size_8_rank_0.pt")
        sys.exit(1)

    world_size = len(shard_files)
    print(f"Found {world_size} shard files in {ckpt_path}")

    # --- Inspect first shard to determine format ---
    first_shard = inspect_shard(str(shard_files[0]))
    first_key = list(first_shard.keys())[0]
    first_val = first_shard[first_key]

    is_sharded_tensor = hasattr(first_val, "local_shards")
    is_flat_tensor = isinstance(first_val, torch.Tensor)

    if is_sharded_tensor:
        print("\nDetected format: ShardedTensor (standard verl FSDP)")
    elif is_flat_tensor:
        print("\nDetected format: Plain tensor shards")
    else:
        print(f"\nERROR: Unknown shard format. First value type: {type(first_val)}")
        sys.exit(1)

    # --- Load base model for parameter shapes and names ---
    print(f"\nLoading base model from {base_model_path} for parameter shapes...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    param_shapes = OrderedDict()
    for name, param in model.named_parameters():
        param_shapes[name] = param.shape

    print(f"Base model has {len(param_shapes)} parameters")

    # --- Load all shard files ---
    print(f"\nLoading all {world_size} shard files...")
    all_shards = [first_shard]
    for i in range(1, world_size):
        print(f"  Loading rank {i}...")
        shard = torch.load(str(shard_files[i]), map_location="cpu", weights_only=False)
        all_shards.append(shard)

    # --- Check if shard keys match model parameter names ---
    shard_keys = set(first_shard.keys())
    model_keys = set(param_shapes.keys())

    if shard_keys == model_keys:
        print("Shard keys match model parameter names exactly.")
        key_mapping = {k: k for k in shard_keys}
    elif all(k.startswith("model.") for k in shard_keys) and not all(k.startswith("model.") for k in model_keys):
        # Shard keys have "model." prefix but model keys don't
        print("Shard keys have 'model.' prefix - will strip it.")
        key_mapping = {k: k.replace("model.", "", 1) for k in shard_keys}
    elif all(not k.startswith("model.") for k in shard_keys) and all(k.startswith("model.") for k in model_keys):
        # Model keys have "model." prefix but shard keys don't
        print("Model keys have 'model.' prefix - will add it.")
        key_mapping = {k: f"model.{k}" for k in shard_keys}
    else:
        # Try to match by finding common patterns
        print("WARNING: Key names don't match directly. Attempting fuzzy matching...")
        key_mapping = {}
        for sk in shard_keys:
            # Strip common prefixes
            clean = sk
            for prefix in ["_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "module."]:
                clean = clean.replace(prefix, "")
            if clean in model_keys:
                key_mapping[sk] = clean
            else:
                print(f"  WARNING: No match for shard key: {sk}")
                key_mapping[sk] = clean  # Use cleaned name anyway

    # --- Merge shards ---
    print(f"\nMerging {len(first_shard)} parameters...")
    merged_state_dict = OrderedDict()
    errors = []

    for i, shard_key in enumerate(first_shard.keys()):
        model_key = key_mapping.get(shard_key, shard_key)
        target_shape = param_shapes.get(model_key)

        if target_shape is None:
            print(f"  WARNING: Skipping {shard_key} (no matching model parameter)")
            continue

        try:
            if is_sharded_tensor:
                full_tensor = merge_sharded_tensors(all_shards, shard_key)
                # Verify shape
                if full_tensor.shape != target_shape:
                    full_tensor = full_tensor.reshape(target_shape)
            else:
                full_tensor = merge_flat_tensors(all_shards, shard_key, target_shape)

            merged_state_dict[model_key] = full_tensor.to(torch.bfloat16)

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(first_shard)}] {model_key}: {full_tensor.shape}")

        except Exception as e:
            errors.append((shard_key, str(e)))
            print(f"  ERROR merging {shard_key}: {e}")

    if errors:
        print(f"\nWARNING: {len(errors)} parameters failed to merge:")
        for key, err in errors:
            print(f"  {key}: {err}")

    # --- Load merged state dict into model ---
    print(f"\nLoading merged state dict into model ({len(merged_state_dict)} parameters)...")
    missing, unexpected = model.load_state_dict(merged_state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    {k}")

    # --- Save as HuggingFace format ---
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving merged model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)

    # Copy tokenizer (prefer from checkpoint's huggingface/ dir, fallback to base model)
    hf_dir = ckpt_path / "huggingface"
    tokenizer_source = str(hf_dir) if hf_dir.exists() else base_model_path
    print(f"Saving tokenizer from {tokenizer_source}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.save_pretrained(str(output_path))

    print(f"\nDone! Merged model saved to {output_path}")
    print(f"You can now load with vLLM: --model {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge verl FSDP sharded checkpoint to HuggingFace format")
    parser.add_argument(
        "--ckpt_dir", type=str, required=True,
        help="Path to the actor/ directory containing model shard files",
    )
    parser.add_argument(
        "--base_model_path", type=str, required=True,
        help="Path to the base HuggingFace model (for architecture and parameter shapes)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Where to save the merged HuggingFace model",
    )
    args = parser.parse_args()

    merge_checkpoint(args.ckpt_dir, args.base_model_path, args.output_dir)
