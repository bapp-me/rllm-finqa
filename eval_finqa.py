"""
FinQA Model Evaluation Script

Evaluates trained FinQA models on a custom test set and logs results to SwanLab.
Produces per-episode metrics identical to validation, plus detailed JSONL output.

Usage:
    python -m projects.finqa.eval_finqa \
        --model_name finqa-grpo-incur \
        --model_path /path/to/checkpoint \
        --base_url http://localhost:30000/v1 \
        --output_dir ./eval_results

Environment variables for judge model:
    FINQA_JUDGE_API_TYPE=chat_completions
    FINQA_JUDGE_BASE_URL=http://localhost:8000/v1
    FINQA_JUDGE_API_KEY=None
    FINQA_JUDGE_MODEL=/path/to/judge/model
"""

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from projects.finqa.fin_qa_agent import FinQAAgent
from projects.finqa.fin_qa_environment import FinQAEnvironment
from projects.finqa.prepare_eval_data import prepare_eval_data
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.utils.tracking import Tracking


def extract_episode_metrics(trajectory, task: dict) -> dict:
    """
    Extract per-episode metrics from a trajectory, replicating
    FinQAWorkflow.collect_metrics() logic from train_finqa.py:88-129.
    """
    metrics = {}

    # Get metadata from the last step's info (set by reward function)
    metadata = {}
    is_correct = False
    if trajectory.steps:
        last_step = trajectory.steps[-1]
        metadata = last_step.info.get("metadata", {})
        is_correct_val = last_step.info.get("is_correct")
        if is_correct_val is not None:
            is_correct = bool(is_correct_val)
        elif trajectory.reward is not None:
            is_correct = trajectory.reward > 0

    # Copy all metadata into metrics (same as episode.metrics.update(metadata))
    metrics.update(metadata)
    metrics["is_correct"] = is_correct

    reward_value = float(metadata.get("correctness_reward", 0.0))
    is_correct_value = 1.0 if is_correct else 0.0
    num_tool_calls = float(metadata.get("num_tool_calls", 0.0))

    # Sparse grouped metrics (same as collect_metrics)
    if float(metadata.get("is_single_table_sample", 0.0)) >= 0.5:
        metrics["reward_single_table"] = reward_value
        metrics["pass_single_table"] = is_correct_value
        metrics["steps_single_table"] = num_tool_calls

    if float(metadata.get("is_multi_table_sample", 0.0)) >= 0.5:
        metrics["reward_multi_table"] = reward_value
        metrics["pass_multi_table"] = is_correct_value
        metrics["steps_multi_table"] = num_tool_calls

    if float(metadata.get("is_multi_table_medium_sample", 0.0)) >= 0.5:
        metrics["reward_multi_table_medium"] = reward_value
        metrics["pass_multi_table_medium"] = is_correct_value

    if float(metadata.get("is_multi_table_hard_sample", 0.0)) >= 0.5:
        metrics["reward_multi_table_hard"] = reward_value
        metrics["pass_multi_table_hard"] = is_correct_value

    if float(metadata.get("is_negative_single_table_sample", 0.0)) >= 0.5:
        metrics["reward_negative_single_table"] = reward_value
        metrics["pass_negative_single_table"] = is_correct_value
        metrics["steps_negative_single_table"] = float(metadata.get("negative_single_table_num_steps", 0.0))

    if float(metadata.get("is_negative_multi_table_sample", 0.0)) >= 0.5:
        metrics["reward_negative_multi_table"] = reward_value
        metrics["pass_negative_multi_table"] = is_correct_value
        metrics["steps_negative_multi_table"] = float(metadata.get("negative_multi_table_num_steps", 0.0))

    return metrics


def aggregate_metrics(all_metrics: list[dict], all_tasks: list[dict]) -> dict:
    """
    Aggregate per-episode metrics by data_source, replicating the
    _validate_agent() logic from agent_workflow_trainer.py:586-603.
    """
    aggregated = {}

    # Group by data_source
    source_metrics = defaultdict(lambda: defaultdict(list))
    source_correct = defaultdict(list)

    for metrics, task in zip(all_metrics, all_tasks):
        data_source = task.get("data_source", "unknown")
        is_correct = metrics.get("is_correct", False)
        source_correct[data_source].append(1.0 if is_correct else 0.0)

        for key, value in metrics.items():
            if key == "is_correct":
                continue
            try:
                source_metrics[data_source][key].append(float(value))
            except (ValueError, TypeError):
                continue

    # Compute per-source aggregated metrics
    for data_source in sorted(source_correct.keys()):
        correct_list = source_correct[data_source]
        aggregated[f"eval/{data_source}/pass@1"] = sum(correct_list) / len(correct_list) if correct_list else 0.0
        aggregated[f"eval/{data_source}/count"] = len(correct_list)

        # Add workflow metrics for this data source
        for key, values in source_metrics[data_source].items():
            if values:
                aggregated[f"eval/{data_source}/{key}"] = sum(values) / len(values)

    # Overall metrics
    all_correct = []
    for correct_list in source_correct.values():
        all_correct.extend(correct_list)
    if all_correct:
        aggregated["eval/overall/pass@1"] = sum(all_correct) / len(all_correct)
        aggregated["eval/overall/count"] = len(all_correct)

    return aggregated


def get_data_type(task: dict) -> str:
    """Get human-readable data type label for a task."""
    cs = task.get("curriculum_stage", "")
    if cs.startswith("negative_"):
        return cs  # negative_single_table or negative_multi_table
    if cs == "single_table":
        return "single_table"
    if cs in ("multi_table_medium", "multi_table_easy"):
        return "multi_table_medium"
    if cs == "multi_table_hard":
        return "multi_table_hard"
    return cs or "unknown"


def extract_conversation(trajectory) -> list[dict]:
    """Extract the full conversation from trajectory steps."""
    if trajectory.steps:
        # The last step's chat_completions contains the full conversation
        last_step = trajectory.steps[-1]
        if hasattr(last_step, "chat_completions") and last_step.chat_completions:
            return last_step.chat_completions
    return []


def build_jsonl_record(trajectory, task: dict, metrics: dict) -> dict:
    """Build a single JSONL record for detailed output."""
    metadata = {}
    if trajectory.steps:
        metadata = trajectory.steps[-1].info.get("metadata", {})

    # Multi-table rubric scores
    rubric = {}
    if float(metadata.get("is_multi_table_sample", 0.0)) >= 0.5:
        for key in ["primary_data_score", "derived_metrics_score", "reasoning_score", "consistency_score"]:
            rubric_key = f"multi_table_{key}"
            if rubric_key in metadata:
                rubric[key] = metadata[rubric_key]
        if "multi_table_overall_score" in metadata:
            rubric["overall_score"] = metadata["multi_table_overall_score"]

    record = {
        "data_type": get_data_type(task),
        "question": task.get("question", ""),
        "core_question": task.get("core_question", ""),
        "ground_truth": task.get("ground_truth", ""),
        "company": task.get("company", ""),
        "question_type": task.get("question_type", ""),
        "question_id": task.get("question_id", ""),
        "is_correct": metrics.get("is_correct", False),
        "reward": float(metadata.get("correctness_reward", 0.0)),
        "rubric_scores": rubric if rubric else None,
        "right_table_access_reward": float(metadata.get("right_table_access_reward", 0.0)),
        "num_tool_calls": float(metadata.get("num_tool_calls", 0.0)),
        "num_steps": len(trajectory.steps),
        "tool_stats": {},
        "judge_stats": {
            "judge_request_ok": float(metadata.get("judge_request_ok", 0.0)),
            "judge_client_unavailable": float(metadata.get("judge_client_unavailable", 0.0)),
            "judge_timeout_error": float(metadata.get("judge_timeout_error", 0.0)),
            "judge_api_error": float(metadata.get("judge_api_error", 0.0)),
            "judge_parse_error": float(metadata.get("judge_parse_error", 0.0)),
        },
        "conversation": extract_conversation(trajectory),
    }

    # Extract per-tool stats
    prefix = get_data_type(task).replace("_medium", "").replace("_hard", "")
    for tool_name in ["sql_query", "get_table_info", "get_table_names", "calculator", "format_error"]:
        sr_key = f"{prefix}_{tool_name}_success_rate"
        calls_key = f"{prefix}_{tool_name}_calls"
        if sr_key in metadata or calls_key in metadata:
            record["tool_stats"][tool_name] = {
                "success_rate": float(metadata.get(sr_key, 0.0)),
                "calls": float(metadata.get(calls_key, 0.0)),
            }

    return record


def run_evaluation(
    model_name: str,
    model_path: str,
    base_url: str,
    output_dir: str,
    tokenizer_path: str | None = None,
    vllm_model_name: str | None = None,
    project_name: str = "finqa-eval",
    n_parallel: int = 50,
    max_steps: int = 20,
    max_prompt_length: int = 4096,
    max_response_length: int = 16384,
):
    """Run evaluation for a single model.

    Args:
        model_path: Path to model (used for tokenizer if tokenizer_path not set)
        tokenizer_path: Separate tokenizer path (for checkpoints where tokenizer != model weights)
        vllm_model_name: Model name as known by the vLLM server (defaults to model_path)
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    tokenizer_path = tokenizer_path or model_path
    vllm_model_name = vllm_model_name or model_path

    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Tokenizer path: {tokenizer_path}")
    print(f"vLLM model name: {vllm_model_name}")
    print(f"API endpoint: {base_url}")
    print(f"{'='*60}\n")

    # --- 1. Load tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # --- 2. Build eval dataset ---
    print("Building evaluation dataset...")
    eval_data = prepare_eval_data()

    # --- 3. Setup execution engine ---
    sampling_params = {"temperature": 0.6, "top_p": 0.95}
    engine = AgentExecutionEngine(
        agent_class=FinQAAgent,
        env_class=FinQAEnvironment,
        engine_name="openai",
        rollout_engine_args={
            "model": vllm_model_name,
            "base_url": base_url,
            "api_key": os.getenv("FINQA_MODEL_API_KEY", "None"),
        },
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        n_parallel_agents=n_parallel,
        max_steps=max_steps,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
    )

    # --- 4. Run inference ---
    print(f"Running inference on {len(eval_data)} samples with {n_parallel} parallel agents...")
    start_time = time.time()
    results = asyncio.run(engine.execute_tasks(eval_data))
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.1f}s ({inference_time/len(eval_data):.1f}s per sample)")

    # --- 5. Compute per-episode metrics ---
    print("Computing metrics...")
    all_metrics = []
    all_records = []

    for trajectory, task in zip(results, eval_data):
        metrics = extract_episode_metrics(trajectory, task)
        all_metrics.append(metrics)

        record = build_jsonl_record(trajectory, task, metrics)
        all_records.append(record)

    # --- 6. Aggregate metrics ---
    aggregated = aggregate_metrics(all_metrics, eval_data)
    aggregated["eval/inference_time_s"] = inference_time
    aggregated["eval/samples_per_second"] = len(eval_data) / inference_time

    # --- 7. Log to SwanLab ---
    print("Logging to SwanLab...")
    logger = Tracking(
        project_name=project_name,
        experiment_name=model_name,
        default_backend=["console", "swanlab"],
    )
    logger.log(data=aggregated, step=0)
    logger.finish()

    # --- 8. Save detailed JSONL output ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    jsonl_file = output_path / f"{model_name}_results.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(f"Detailed results saved to {jsonl_file}")

    # Save aggregated metrics as JSON
    metrics_file = output_path / f"{model_name}_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False, default=str)
    print(f"Aggregated metrics saved to {metrics_file}")

    # --- 9. Print summary ---
    print(f"\n{'='*60}")
    print(f"Results for {model_name}:")
    print(f"{'='*60}")
    for key in sorted(aggregated.keys()):
        if "pass@1" in key or "count" in key:
            val = aggregated[key]
            if "pass@1" in key:
                print(f"  {key}: {val:.4f} ({val*100:.1f}%)")
            else:
                print(f"  {key}: {val}")
    print()

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="FinQA Model Evaluation")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model (used for logging)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer path (defaults to model_path)")
    parser.add_argument("--vllm_model_name", type=str, default=None, help="Model name as vLLM knows it (defaults to model_path)")
    parser.add_argument("--base_url", type=str, default="http://localhost:30000/v1", help="vLLM server URL")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Directory for output files")
    parser.add_argument("--project_name", type=str, default="finqa-eval", help="SwanLab project name")
    parser.add_argument("--n_parallel", type=int, default=50, help="Number of parallel agents")
    parser.add_argument("--max_steps", type=int, default=20, help="Max ReAct steps per episode")
    parser.add_argument("--max_prompt_length", type=int, default=4096, help="Max prompt length in tokens")
    parser.add_argument("--max_response_length", type=int, default=16384, help="Max response length in tokens")
    args = parser.parse_args()

    run_evaluation(
        model_name=args.model_name,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        vllm_model_name=args.vllm_model_name,
        base_url=args.base_url,
        output_dir=args.output_dir,
        project_name=args.project_name,
        n_parallel=args.n_parallel,
        max_steps=args.max_steps,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
    )


if __name__ == "__main__":
    main()
