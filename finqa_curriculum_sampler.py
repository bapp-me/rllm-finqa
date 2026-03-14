"""
In-training curriculum learning sampler for FinQA.

Dynamically adjusts sampling weights across difficulty levels (data_source)
based on reward improvement velocity and saturation detection.

Key design choices for FinQA's heterogeneous reward distributions:
- single_table: discrete {0, 1} reward → uses raw success rate
- multi_table: continuous [0, 1] reward → binarized with threshold for success rate

Exploitation signal: reward improvement velocity (Δ success_rate per step),
NOT |advantage| mean — avoids cross-source incomparability due to different
reward distributions.

Saturation detection: when a source's success rate exceeds a high-water mark
(e.g. 0.85), its weight is dampened to redirect training resources to sources
with more room for improvement.

Usage in train_finqa.sh:
    data.sampler.class_path=pkg://projects.finqa.finqa_curriculum_sampler
    data.sampler.class_name=FinQACurriculumSampler
    data.dataloader_num_workers=0
"""

import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Sampler

from verl import DataProto

try:
    from verl.experimental.dataset.sampler import AbstractCurriculumSampler
except ImportError:
    class AbstractCurriculumSampler(Sampler):
        def update(self, batch: "DataProto") -> None: ...

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-source tracker
# ---------------------------------------------------------------------------

@dataclass
class SourceTracker:
    """Track success rate trajectory for one data source."""
    # For success rate: binarized reward
    success_threshold: float = 1.0  # single_table: >=1.0; multi_table: will be overridden
    n_samples: int = 0

    # Sliding window of binarized outcomes (1=success, 0=failure)
    window_size: int = 128
    recent_outcomes: deque = field(default_factory=lambda: deque(maxlen=128))

    # Track success rate history for velocity computation (one entry per update call)
    rate_history: list[float] = field(default_factory=list)
    max_history: int = 20  # keep last N rate snapshots

    def update(self, rewards: np.ndarray) -> None:
        if len(rewards) == 0:
            return
        outcomes = (rewards >= self.success_threshold).astype(float)
        for o in outcomes:
            self.recent_outcomes.append(o)
        self.n_samples += len(rewards)

        # Snapshot current rate
        rate = self.success_rate
        self.rate_history.append(rate)
        if len(self.rate_history) > self.max_history:
            self.rate_history = self.rate_history[-self.max_history:]

    @property
    def success_rate(self) -> float:
        if not self.recent_outcomes:
            return 0.0
        return sum(self.recent_outcomes) / len(self.recent_outcomes)

    @property
    def improvement_velocity(self) -> float:
        """Reward improvement speed: positive = getting better, zero/negative = saturated."""
        if len(self.rate_history) < 2:
            return 0.0
        # Compare recent half vs earlier half
        mid = len(self.rate_history) // 2
        if mid == 0:
            return 0.0
        early = np.mean(self.rate_history[:mid])
        recent = np.mean(self.rate_history[mid:])
        return float(recent - early)


# ---------------------------------------------------------------------------
# Curriculum controller
# ---------------------------------------------------------------------------

class CurriculumController:
    def __init__(
        self,
        data_sources: list[str],
        source_counts: dict[str, int],
        temperature: float = 0.5,
        saturation_threshold: float = 0.75,
        multi_table_success_threshold: float = 0.8,
        min_weight: float = 0.1,
    ):
        self.data_sources = sorted(set(data_sources))
        self.temperature = temperature
        self.saturation_threshold = saturation_threshold
        self.min_weight = min_weight  # minimum weight for any source

        self.trackers: dict[str, SourceTracker] = {}
        for s in self.data_sources:
            threshold = multi_table_success_threshold if "multi" in s else 1.0
            self.trackers[s] = SourceTracker(
                success_threshold=threshold,
                window_size=128,
                recent_outcomes=deque(maxlen=128),
            )

        # Initial weights proportional to data counts
        total = sum(source_counts.get(s, 1) for s in self.data_sources)
        self._initial_weights = {s: source_counts.get(s, 1) / total for s in self.data_sources}

    def compute_sampling_weights(self) -> dict[str, float]:
        total_samples = max(1, sum(t.n_samples for t in self.trackers.values()))

        # Check if we have enough data to compute meaningful signals
        min_samples = 32  # need at least this many samples per source
        has_enough_data = all(t.n_samples >= min_samples for t in self.trackers.values())

        if not has_enough_data:
            # Early phase: use data-proportional weights
            return dict(self._initial_weights)

        scores = {}
        for source in self.data_sources:
            tracker = self.trackers[source]

            # Exploitation: improvement velocity (how fast is this source improving?)
            velocity = max(0.0, tracker.improvement_velocity)

            # Exploration: UCB bonus for under-sampled sources
            exploration = np.sqrt(2 * np.log(total_samples + 1) / (tracker.n_samples + 1))

            # Room for improvement: 1 - success_rate (higher = more room)
            room = 1.0 - tracker.success_rate

            # Combined score: prioritize sources that are (a) improving fast AND
            # (b) still have room to improve. Exploration ensures we don't ignore any source.
            scores[source] = (velocity + 0.1) * room + exploration

            # Saturation dampening: if success rate is very high, reduce weight
            if tracker.success_rate >= self.saturation_threshold:
                scores[source] *= 0.3  # strong dampening

        # Softmax with temperature
        source_list = self.data_sources
        score_arr = np.array([scores[s] for s in source_list], dtype=np.float64)
        score_arr -= score_arr.max()
        exp_scores = np.exp(score_arr / self.temperature)
        weights = exp_scores / exp_scores.sum()

        result = {}
        for s, w in zip(source_list, weights):
            # Enforce minimum weight so no source is completely starved
            result[s] = max(float(w), self.min_weight)

        # Re-normalize after applying min_weight
        w_sum = sum(result.values())
        result = {s: v / w_sum for s, v in result.items()}
        return result


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class FinQACurriculumSampler(AbstractCurriculumSampler):
    """
    Curriculum sampler for FinQA with scene-specific adaptations:

    1. Initial weights proportional to data counts (not uniform)
    2. Exploitation signal = reward improvement velocity (not |advantage|)
    3. Multi-table rewards binarized at threshold 0.6 for success rate tracking
    4. Saturation detection: sources with success_rate > 0.85 get dampened
    5. Minimum weight floor prevents any source from being starved
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
                source = item.get("data_source", None)
                if source is None and isinstance(item.get("extra_info"), dict):
                    source = item["extra_info"].get("data_source", "unknown")
                source = source or "unknown"
            self.source_indices[source].append(i)

        sources = list(self.source_indices.keys())
        counts = {s: len(idxs) for s, idxs in self.source_indices.items()}
        logger.info(f"[FinQACurriculumSampler] data_source counts: {counts}")
        print(f"[FinQACurriculumSampler] data_source counts: {counts}")

        self.controller = CurriculumController(
            data_sources=sources,
            source_counts=counts,
        )
        # Initial weights proportional to data counts
        self._weights = dict(self.controller._initial_weights)

    # ---- AbstractCurriculumSampler interface ----

    def update(self, batch: DataProto) -> None:
        """Called after each training step with the processed batch."""
        data_sources = batch.non_tensor_batch.get("data_source", None)
        if data_sources is None:
            return

        rewards = batch.batch.get("token_level_rewards", None)
        if rewards is None:
            return
        response_mask = batch.batch.get("response_mask", None)

        # Collect per-sample total reward grouped by source
        source_rewards: dict[str, list[float]] = defaultdict(list)
        for i, source in enumerate(data_sources):
            source = str(source)
            if response_mask is not None:
                sample_reward = float((rewards[i] * response_mask[i]).sum().item())
            else:
                sample_reward = float(rewards[i].sum().item())
            source_rewards[source].append(sample_reward)

        # Update trackers
        for source, reward_list in source_rewards.items():
            if source in self.controller.trackers:
                self.controller.trackers[source].update(np.array(reward_list))

        # Recompute weights
        self._weights = self.controller.compute_sampling_weights()

        # Log
        stats = []
        for s in self.controller.data_sources:
            t = self.controller.trackers[s]
            w = self._weights.get(s, 0)
            stats.append(
                f"{s}: w={w:.3f} rate={t.success_rate:.3f} "
                f"vel={t.improvement_velocity:+.4f} n={t.n_samples}"
            )
        msg = f"[Curriculum] {'; '.join(stats)}"
        logger.info(msg)
        print(msg)

    # ---- Sampler interface ----

    def __iter__(self):
        sources = list(self._weights.keys())
        weights = [self._weights[s] for s in sources]

        valid = [(s, w) for s, w in zip(sources, weights) if self.source_indices.get(s)]
        if not valid:
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
