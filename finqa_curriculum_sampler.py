"""
In-training curriculum learning sampler for FinQA.

Uses Bayesian learnability estimation with UCB exploration to dynamically
adjust sampling weights across difficulty levels (data_source) during training.

Adapted from DUMP project's curriculum learning approach for use with
verl's AbstractCurriculumSampler interface.

Usage in train_finqa.sh:
    data.sampler.class_path=projects.finqa.finqa_curriculum_sampler
    data.sampler.class_name=FinQACurriculumSampler
    data.dataloader_num_workers=0
"""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Sampler

from verl import DataProto

try:
    from verl.experimental.dataset.sampler import AbstractCurriculumSampler
except ImportError:
    # Fallback: define a minimal base so the file still imports
    class AbstractCurriculumSampler(Sampler):
        def update(self, batch: "DataProto") -> None: ...

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learnability estimator (per data_source)
# ---------------------------------------------------------------------------

@dataclass
class LearnabilityEstimator:
    """Track success rate and advantage magnitude for one data source."""
    alpha: float = 1.0          # Beta distribution (successes)
    beta_param: float = 1.0     # Beta distribution (failures)
    mu: float = 0.0             # Running mean of |advantages|
    n_samples: int = 0
    window_size: int = 128
    recent_rewards: list[float] = field(default_factory=list)
    recent_advantages: list[float] = field(default_factory=list)

    def update(self, rewards: np.ndarray, advantages: np.ndarray) -> None:
        if len(rewards) == 0:
            return
        self.recent_rewards.extend(rewards.tolist())
        self.recent_advantages.extend(advantages.tolist())
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards = self.recent_rewards[-self.window_size:]
        if len(self.recent_advantages) > self.window_size:
            self.recent_advantages = self.recent_advantages[-self.window_size:]

        successes = int(np.sum(rewards >= 1))
        failures = len(rewards) - successes
        self.alpha += successes
        self.beta_param += failures
        self.mu = float(np.mean(np.abs(self.recent_advantages)))
        self.n_samples += len(rewards)


# ---------------------------------------------------------------------------
# Curriculum controller (manages all sources)
# ---------------------------------------------------------------------------

class CurriculumController:
    def __init__(self, data_sources: list[str], temperature: float = 0.5):
        self.data_sources = sorted(set(data_sources))
        self.temperature = temperature
        self.estimators: dict[str, LearnabilityEstimator] = {
            s: LearnabilityEstimator() for s in self.data_sources
        }

    def compute_sampling_weights(self) -> dict[str, float]:
        total_samples = max(1, sum(e.n_samples for e in self.estimators.values()))
        scores = []
        for source in self.data_sources:
            est = self.estimators[source]
            exploration = np.sqrt(2 * np.log(total_samples + 1) / (est.n_samples + 1))
            scores.append(est.mu + exploration)
        scores = np.array(scores, dtype=np.float64)
        # Softmax with temperature
        scores -= scores.max()
        exp_scores = np.exp(scores / self.temperature)
        weights = exp_scores / exp_scores.sum()
        return {s: float(w) for s, w in zip(self.data_sources, weights)}


# ---------------------------------------------------------------------------
# Sampler implementing verl's AbstractCurriculumSampler
# ---------------------------------------------------------------------------

class FinQACurriculumSampler(AbstractCurriculumSampler):
    """
    Curriculum sampler for FinQA that dynamically adjusts sampling weights
    across data sources (single_table, multi_table, etc.) based on training
    performance signals.

    Required config:
        data.sampler.class_path = "projects.finqa.finqa_curriculum_sampler"
        data.sampler.class_name = "FinQACurriculumSampler"
        data.dataloader_num_workers = 0
    """

    def __init__(self, data_source: "Sized", data_config: DictConfig):
        self.dataset = data_source
        self.data_config = data_config
        self.seed = data_config.get("seed", 42)
        self.rng = random.Random(self.seed)

        # Group indices by data_source
        self.source_indices: dict[str, list[int]] = defaultdict(list)
        for i in range(len(data_source)):
            item = data_source[i]
            source = "unknown"
            if isinstance(item, dict):
                # data_source may be at top level or nested in extra_info
                source = item.get("data_source", None)
                if source is None and isinstance(item.get("extra_info"), dict):
                    source = item["extra_info"].get("data_source", "unknown")
                source = source or "unknown"
            self.source_indices[source].append(i)

        sources = list(self.source_indices.keys())
        counts = {s: len(idxs) for s, idxs in self.source_indices.items()}
        logger.info(f"[FinQACurriculumSampler] data_source counts: {counts}")
        print(f"[FinQACurriculumSampler] data_source counts: {counts}")

        self.controller = CurriculumController(data_sources=sources)
        # Initialize weights uniformly
        self._weights = {s: 1.0 / len(sources) for s in sources}

    # ---- AbstractCurriculumSampler interface ----

    def update(self, batch: DataProto) -> None:
        """Called by the trainer after each training step with the processed batch."""
        data_sources = batch.non_tensor_batch.get("data_source", None)
        if data_sources is None:
            return

        # Collect rewards and advantages per source
        source_rewards: dict[str, list[float]] = defaultdict(list)
        source_advantages: dict[str, list[float]] = defaultdict(list)

        rewards = batch.batch.get("token_level_rewards", None)
        advantages = batch.batch.get("advantages", None)
        if rewards is None or advantages is None:
            return

        response_mask = batch.batch.get("response_mask", None)

        for i, source in enumerate(data_sources):
            source = str(source)
            sample_reward = float(rewards[i].sum().item())
            if response_mask is not None:
                mask = response_mask[i].bool()
                adv_vals = advantages[i][mask]
                sample_adv = float(adv_vals.mean().item()) if adv_vals.numel() > 0 else 0.0
            else:
                sample_adv = float(advantages[i].mean().item())
            source_rewards[source].append(sample_reward)
            source_advantages[source].append(sample_adv)

        # Update estimators
        for source in source_rewards:
            if source in self.controller.estimators:
                self.controller.estimators[source].update(
                    np.array(source_rewards[source]),
                    np.array(source_advantages[source]),
                )

        # Recompute weights
        self._weights = self.controller.compute_sampling_weights()

        # Log current weights
        stats = []
        for s in self.controller.data_sources:
            est = self.controller.estimators[s]
            w = self._weights.get(s, 0)
            rate = est.alpha / (est.alpha + est.beta_param)
            stats.append(f"{s}: w={w:.3f} rate={rate:.3f} adv={est.mu:.4f} n={est.n_samples}")
        logger.info(f"[Curriculum] weights updated: {'; '.join(stats)}")
        print(f"[Curriculum] weights updated: {'; '.join(stats)}")

    # ---- Sampler interface ----

    def __iter__(self):
        sources = list(self._weights.keys())
        weights = [self._weights[s] for s in sources]

        # Filter to sources with actual indices
        valid = [(s, w) for s, w in zip(sources, weights) if self.source_indices.get(s)]
        if not valid:
            # Fallback: yield all indices shuffled
            all_indices = list(range(len(self.dataset)))
            self.rng.shuffle(all_indices)
            yield from all_indices
            return

        valid_sources, valid_weights = zip(*valid)
        w_sum = sum(valid_weights)
        valid_weights = [w / w_sum for w in valid_weights]

        for _ in range(len(self.dataset)):
            source = self.rng.choices(valid_sources, weights=valid_weights, k=1)[0]
            idx = self.rng.choice(self.source_indices[source])
            yield idx

    def __len__(self):
        return len(self.dataset)
