"""
Evaluate FinQA models using the verl training infrastructure.

Instead of merging FSDP checkpoints, this uses verl's native checkpoint loading
with val_only mode. The validation dataset is replaced with a custom test set.

Usage:
    python3 -m projects.finqa.eval_finqa_verl \
        trainer.default_local_dir=/path/to/checkpoint_dir \
        trainer.experiment_name=model-name \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.resume_mode=auto \
        ...
"""

import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer

from .fin_qa_agent import FinQAAgent
from .fin_qa_environment import FinQAEnvironment
from .prepare_eval_data import prepare_eval_data
from .train_finqa import FinQAWorkflow


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config):
    # Replace val split with custom test set (250 single + 50 multi + 40 negative)
    test_data = prepare_eval_data()
    DatasetRegistry.register_dataset("finqa", test_data, "val")

    train_dataset = DatasetRegistry.load_dataset("finqa", "train")
    val_dataset = DatasetRegistry.load_dataset("finqa", "val")

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
