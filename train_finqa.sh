set -x

export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

TOTAL_EPOCHS=${TOTAL_EPOCHS:-6}
MAX_HARD_TRAIN=${MAX_HARD_TRAIN:-400}
SEED=${SEED:-42}

for EPOCH in $(seq 1 "$TOTAL_EPOCHS"); do
    if [ "$EPOCH" -eq 1 ]; then
        RESUME_MODE=disable
    else
        RESUME_MODE=auto
    fi

    python3 -m projects.finqa.prepare_finqa_curriculum_data \
        --epoch "$EPOCH" \
        --total-epochs "$TOTAL_EPOCHS" \
        --max-hard-train "$MAX_HARD_TRAIN" \
        --seed "$SEED"

    python3 -m projects.finqa.train_finqa \
        algorithm.adv_estimator=grpo \
        data.train_batch_size=384 \
        data.val_batch_size=648 \
        data.max_prompt_length=3072 \
        data.max_response_length=16384 \
        actor_rollout_ref.model.path=/home/qinhengyi/rllm/projects/finqa/qwen \
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
        actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
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
        trainer.logger=['console','swanlab'] \
        trainer.project_name='finqa-grpo-curriculum' \
        trainer.experiment_name='finqa-grpo-test2' \
        trainer.val_before_train=False \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.default_hdfs_dir=null \
        trainer.resume_mode=$RESUME_MODE \
        rllm.agent.max_steps=20 \
        rllm.stepwise_advantage.enable=False \
        rllm.workflow.n_parallel_tasks=1536 \
        trainer.total_epochs=1
done