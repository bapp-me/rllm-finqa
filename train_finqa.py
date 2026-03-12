import hydra
from collections import Counter

from rllm.agents.agent import Episode
from rllm.data.dataset import DatasetRegistry
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow
from rllm.workflows.workflow import TerminationEvent, TerminationReason

from .fin_qa_agent import FinQAAgent
from .fin_qa_environment import FinQAEnvironment


class FinQAWorkflow(MultiTurnWorkflow):
    """MultiTurnWorkflow with reward logging"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        observation, info = await self.run_in_executor(self.reset, task=task, uid=uid)

        self.agent.update_from_env(observation, 0, False, info)

        # Calculate max context once (max_prompt_length + max_response_length)
        max_model_len = self.rollout_engine.max_prompt_length + self.rollout_engine.max_response_length
        min_response_buffer = 1000  # Minimum tokens to reserve for model response

        for _ in range(1, self.max_steps + 1):
            # Check if conversation is approaching context limit
            if hasattr(self.rollout_engine, "chat_parser"):
                # verl backend - use chat_parser
                prompt = self.rollout_engine.chat_parser.parse(
                    self.agent.chat_completions,
                    add_generation_prompt=True,
                    is_first_msg=True,
                )
                prompt_length = len(self.rollout_engine.tokenizer.encode(prompt, add_special_tokens=False))
            else:
                # Tinker backend - use tokenizer directly
                prompt_ids = self.rollout_engine.tokenizer.apply_chat_template(
                    self.agent.chat_completions,
                    add_generation_prompt=True,
                    tokenize=True,
                )
                prompt_length = len(prompt_ids)

            if prompt_length > max_model_len - min_response_buffer:
                raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

            output: ModelOutput = await self.rollout_engine.get_model_response(
                self.agent.chat_completions,
                application_id=uid,
                enforce_max_prompt_length=False,
                **kwargs,
            )
            response = output.text

            action = self.agent.update_from_model(response)

            # Store model_output on step for Tinker training (verl uses chat_completions)
            if not hasattr(self.rollout_engine, "chat_parser") and self.agent.trajectory.steps:
                self.agent.trajectory.steps[-1].model_output = output

            next_obs, reward, done, info = await self.run_in_executor(self.env.step, action.action)

            self.agent.update_from_env(next_obs, reward, done, info)

            if output.finish_reason == "length":
                raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

            if done:
                raise TerminationEvent(TerminationReason.ENV_DONE)

        raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)

    def assign_episode_correctness(self, episode: Episode) -> None:
        """Use is_correct from Reward, before relying on default option"""
        # Check if last step has is_correct from environment
        if episode.trajectories and episode.trajectories[0].steps:
            is_correct = episode.trajectories[0].steps[-1].info.get("is_correct")
            if is_correct is not None:
                episode.is_correct = is_correct
                return

        super().assign_episode_correctness(episode)

    def collect_metrics(self, episode: Episode) -> None:
        super().collect_metrics(episode)
        # Added metadata from last step -> for wandb logging
        if episode.trajectories and episode.trajectories[0].steps:
            metadata = episode.trajectories[0].steps[-1].info.get("metadata", {})
            episode.metrics.update(metadata)

            reward_value = float(metadata.get("correctness_reward", 0.0))
            is_correct_value = 1.0 if bool(episode.is_correct) else 0.0

            # Sparse grouped metrics: keys are only emitted for matching samples,
            # so logger aggregation computes per-group means directly.
            if float(metadata.get("is_single_table_sample", 0.0)) >= 0.5:
                episode.metrics["reward_single_table"] = reward_value
                episode.metrics["pass_single_table"] = is_correct_value

            if float(metadata.get("is_multi_table_sample", 0.0)) >= 0.5:
                episode.metrics["reward_multi_table"] = reward_value
                episode.metrics["pass_multi_table"] = is_correct_value

            if float(metadata.get("is_multi_table_medium_sample", 0.0)) >= 0.5:
                episode.metrics["reward_multi_table_medium"] = reward_value
                episode.metrics["pass_multi_table_medium"] = is_correct_value

            if float(metadata.get("is_multi_table_hard_sample", 0.0)) >= 0.5:
                episode.metrics["reward_multi_table_hard"] = reward_value
                episode.metrics["pass_multi_table_hard"] = is_correct_value


def _iter_dataset_examples(dataset):
    """Yield (index, sample) pairs from a dataset-like object."""
    try:
        total = len(dataset)
    except Exception:
        total = None

    if isinstance(total, int):
        for i in range(total):
            try:
                yield i, dataset[i]
            except Exception:
                break
        return

    for i, sample in enumerate(dataset):
        yield i, sample


def _inspect_curriculum_order(dataset, split_name: str, preview_count: int = 20) -> None:
    """Print curriculum ordering diagnostics before training starts."""
    stage_rank = {
        "single_table": 0,
        "multi_table_medium": 1,
        "multi_table_hard": 2,
        "multi_table_other": 3,
    }

    stage_counts = Counter()
    first_indices = {}
    preview = []
    disorder_indices = []
    prev_rank = -1
    total_seen = 0

    for idx, sample in _iter_dataset_examples(dataset):
        if not isinstance(sample, dict):
            continue

        stage = str(sample.get("curriculum_stage") or "single_table")
        stage_counts[stage] += 1
        total_seen += 1

        if stage not in first_indices:
            first_indices[stage] = idx

        if len(preview) < preview_count:
            preview.append(
                {
                    "idx": idx,
                    "stage": stage,
                    "qtype": str(sample.get("question_type") or ""),
                    "qid": str(sample.get("question_id") or ""),
                }
            )

        rank = stage_rank.get(stage, stage_rank["multi_table_other"])
        if rank < prev_rank:
            disorder_indices.append(idx)
            if len(disorder_indices) >= 10:
                break
        prev_rank = rank

    print(f"[{split_name}] Curriculum check: total={total_seen}, stage_counts={dict(stage_counts)}")
    print(f"[{split_name}] First stage indices: {first_indices}")
    print(f"[{split_name}] Preview first {len(preview)} samples:")
    for item in preview:
        print(
            f"[{split_name}] idx={item['idx']} stage={item['stage']} "
            f"qtype={item['qtype']} qid={item['qid']}"
        )

    if disorder_indices:
        print(f"[{split_name}] WARNING: detected stage order violations at indices {disorder_indices}")
    else:
        print(f"[{split_name}] Curriculum order looks monotonic (single -> medium -> hard).")


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("finqa", "train")
    val_dataset = DatasetRegistry.load_dataset("finqa", "val")

    _inspect_curriculum_order(train_dataset, "train")
    _inspect_curriculum_order(val_dataset, "val")

    config.rllm.workflow.use_workflow = True

    trainer = AgentTrainer(
        workflow_class=FinQAWorkflow,
        workflow_args={
            "agent_cls": FinQAAgent,
            "env_cls": FinQAEnvironment,
            "max_steps": 20,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
