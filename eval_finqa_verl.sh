#!/bin/bash
set -x

# ===================================================================
# FinQA Evaluation via verl val_only mode
#
# Uses the training infrastructure to natively load FSDP checkpoints
# and run validation on a custom test set (no checkpoint merging needed).
#
# Prerequisites:
#   - 8 GPUs available (same world_size as training)
#   - ARK_API_KEY set for Doubao judge API
# ===================================================================

export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# ---------- Paths ----------
RLLM_ROOT=${RLLM_ROOT:-/home/qinhengyi/rllm}
CKPT_BASE="${RLLM_ROOT}/checkpoints/finqa-grpo-curriculum"
BASE_MODEL_PATH="${RLLM_ROOT}/projects/finqa/qwen"
OFFICIAL_MODEL_PATH="${RLLM_ROOT}/finqa4b"
EVAL_RESULTS_DIR="${RLLM_ROOT}/eval_results"

# ---------- Judge model (Doubao API, no local GPU needed) ----------
# Uses default Doubao config from fin_qa_reward_doubao.py:
#   API type: responses, Model: doubao-seed-2-0-mini-260215
# Just ensure ARK_API_KEY is set in your environment.

# ---------- Data prep (register train split) ----------
python3 -m projects.finqa.prepare_finqa_data

# ---------- Eval function ----------
run_eval() {
    local model_path="$1"
    local experiment_name="$2"
    local ckpt_dir="$3"
    local resume_mode="$4"

    echo ""
    echo "============================================================"
    echo "Evaluating: ${experiment_name}"
    echo "  Model path: ${model_path}"
    echo "  Checkpoint dir: ${ckpt_dir}"
    echo "  Resume mode: ${resume_mode}"
    echo "============================================================"

    python3 -m projects.finqa.eval_finqa_verl \
        algorithm.adv_estimator=grpo \
        data.train_batch_size=384 \
        data.val_batch_size=648 \
        data.max_prompt_length=3072 \
        data.max_response_length=16384 \
        data.dataloader_num_workers=0 \
        data.sampler.class_path=pkg://projects.finqa.finqa_curriculum_sampler \
        data.sampler.class_name=FinQACurriculumSampler \
        data.neg_single_table_per_batch=40 \
        data.neg_multi_table_per_batch=10 \
        actor_rollout_ref.model.path=${model_path} \
        actor_rollout_ref.hybrid_engine=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=60000 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.mode="async" \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.temperature=0.7 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.92 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.enable_prefix_caching=True \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=81920 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=81920 \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
        actor_rollout_ref.actor.entropy_coeff=0.002 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        rllm.mask_truncated_samples=False \
        trainer.critic_warmup=0 \
        trainer.logger="['console','swanlab']" \
        trainer.project_name='finqa-eval' \
        trainer.experiment_name=${experiment_name} \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=999999 \
        trainer.test_freq=999999 \
        trainer.default_hdfs_dir=null \
        trainer.default_local_dir=${ckpt_dir} \
        trainer.resume_mode=${resume_mode} \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.val_results_dir=${EVAL_RESULTS_DIR} \
        rllm.agent.max_steps=20 \
        rllm.stepwise_advantage.enable=False \
        rllm.workflow.n_parallel_tasks=1152 \
        trainer.total_epochs=1

    echo "Completed evaluation for ${experiment_name}"
}

# ---------- Evaluate checkpoint models ----------
run_eval "${BASE_MODEL_PATH}" "finqa-grpo-incur" "${CKPT_BASE}/finqa-grpo-incur" "auto"
run_eval "${BASE_MODEL_PATH}" "finqa-grpo-neg" "${CKPT_BASE}/finqa-grpo-neg" "auto"
run_eval "${BASE_MODEL_PATH}" "finqa-grpo-nopen" "${CKPT_BASE}/finqa-grpo-nopen" "auto"

# ---------- Evaluate official model (no checkpoint, load directly) ----------
run_eval "${OFFICIAL_MODEL_PATH}" "finqa4b-official" "/tmp/finqa-eval-no-ckpt" "disable"

echo ""
echo "============================================================"
echo "All evaluations complete!"
echo "============================================================"
