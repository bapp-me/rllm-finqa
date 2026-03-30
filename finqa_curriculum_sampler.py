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

Negative samples: fixed number of negative samples are injected per batch
(not subject to curriculum learning). Configured via:
    data.neg_single_table_per_batch=40
    data.neg_multi_table_per_batch=10

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

# Prefix used to identify negative data sources
NEGATIVE_PREFIX = "negative_"


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
    window_size: int = 512
    recent_outcomes: deque = field(default_factory=lambda: deque(maxlen=512))

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
# Curriculum controller (positive sources only)
# ---------------------------------------------------------------------------

class CurriculumController:
    def __init__(
        self,
        data_sources: list[str],
        source_counts: dict[str, int],
        temperature: float = 0.8,
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
                window_size=512,
                recent_outcomes=deque(maxlen=512),
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
            exploration = 0.5 * np.sqrt(2 * np.log(total_samples + 1) / (tracker.n_samples + 1))

            # Room for improvement: 1 - success_rate (higher = more room)
            room = 1.0 - tracker.success_rate

            # Combined score: prioritize sources that are (a) improving fast AND
            # (b) still have room to improve. Exploration ensures we don't ignore any source.
            scores[source] = (velocity + 0.1) * room + exploration

            # Saturation dampening: if success rate is very high, reduce weight smoothly
            if tracker.success_rate >= self.saturation_threshold:
                penalty = (tracker.success_rate - self.saturation_threshold) * 3.0
                multiplier = max(0.4, 1.0 - penalty)
                scores[source] *= multiplier

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
    6. Negative samples: fixed count per batch, not subject to curriculum learning
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

        # Separate positive and negative sources
        self.positive_sources = [s for s in self.source_indices if not s.startswith(NEGATIVE_PREFIX)]
        self.negative_sources = [s for s in self.source_indices if s.startswith(NEGATIVE_PREFIX)]

        all_counts = {s: len(idxs) for s, idxs in self.source_indices.items()}
        positive_counts = {s: all_counts[s] for s in self.positive_sources}
        negative_counts = {s: all_counts[s] for s in self.negative_sources}

        logger.info(f"[FinQACurriculumSampler] positive source counts: {positive_counts}")
        logger.info(f"[FinQACurriculumSampler] negative source counts: {negative_counts}")
        print(f"[FinQACurriculumSampler] positive source counts: {positive_counts}")
        print(f"[FinQACurriculumSampler] negative source counts: {negative_counts}")

        # Curriculum controller only for positive sources
        self.controller = CurriculumController(
            data_sources=self.positive_sources,
            source_counts=positive_counts,
        )

        # Check if user explicitly provided initial_weights in config
        custom_weights = data_config.get("initial_weights", None)
        if custom_weights:
            total_custom = sum(float(v) for v in custom_weights.values())
            for s in self.positive_sources:
                if s in custom_weights:
                    self.controller._initial_weights[s] = float(custom_weights[s]) / total_custom
            logger.info(f"[FinQACurriculumSampler] Overriding initial weights from config: {self.controller._initial_weights}")
            print(f"[FinQACurriculumSampler] Overriding initial weights from config: {self.controller._initial_weights}")

        # Initial weights (positive sources only)
        self._weights = dict(self.controller._initial_weights)

        # Negative sample injection config (per batch)
        self.neg_single_per_batch = int(data_config.get("neg_single_table_per_batch", 40))
        self.neg_multi_per_batch = int(data_config.get("neg_multi_table_per_batch", 10))

        # Batch size for structured batch generation
        self._batch_size = int(data_config.get("gen_batch_size", data_config.get("train_batch_size", 256)))

        # Cap negative counts if batch_size is too small
        total_neg = self.neg_single_per_batch + self.neg_multi_per_batch
        if total_neg >= self._batch_size:
            ratio = (self._batch_size * 0.5) / total_neg
            self.neg_single_per_batch = max(0, int(self.neg_single_per_batch * ratio))
            self.neg_multi_per_batch = max(0, int(self.neg_multi_per_batch * ratio))
            logger.warning(
                f"[FinQACurriculumSampler] Capped negative counts to fit batch_size={self._batch_size}: "
                f"neg_single={self.neg_single_per_batch}, neg_multi={self.neg_multi_per_batch}"
            )

        # Disable negative injection if no negative data available
        if not self.source_indices.get("negative_single_table"):
            if self.neg_single_per_batch > 0:
                logger.warning("[FinQACurriculumSampler] No negative_single_table data found; disabling single negative injection")
            self.neg_single_per_batch = 0
        if not self.source_indices.get("negative_multi_table"):
            if self.neg_multi_per_batch > 0:
                logger.warning("[FinQACurriculumSampler] No negative_multi_table data found; disabling multi negative injection")
            self.neg_multi_per_batch = 0

        total_neg_actual = self.neg_single_per_batch + self.neg_multi_per_batch
        pos_per_batch = self._batch_size - total_neg_actual
        logger.info(
            f"[FinQACurriculumSampler] batch_size={self._batch_size}, "
            f"per batch: {self.neg_single_per_batch} neg_single + "
            f"{self.neg_multi_per_batch} neg_multi + {pos_per_batch} positive"
        )
        print(
            f"[FinQACurriculumSampler] batch_size={self._batch_size}, "
            f"per batch: {self.neg_single_per_batch} neg_single + "
            f"{self.neg_multi_per_batch} neg_multi + {pos_per_batch} positive"
        )

        # Separate trackers for negative sources (for logging, not curriculum)
        self.negative_trackers: dict[str, SourceTracker] = {}
        for s in self.negative_sources:
            self.negative_trackers[s] = SourceTracker(
                success_threshold=1.0,
                window_size=256,
                recent_outcomes=deque(maxlen=256),
            )

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

        # Update positive source trackers (curriculum controller)
        for source, reward_list in source_rewards.items():
            if source in self.controller.trackers:
                self.controller.trackers[source].update(np.array(reward_list))

        # Update negative source trackers (logging only)
        for source, reward_list in source_rewards.items():
            if source in self.negative_trackers:
                self.negative_trackers[source].update(np.array(reward_list))

        # Recompute curriculum weights (positive sources only)
        self._weights = self.controller.compute_sampling_weights()

        # Log positive sources
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

        # Log negative sources
        if self.negative_trackers:
            neg_stats = []
            for s in sorted(self.negative_trackers):
                t = self.negative_trackers[s]
                neg_stats.append(
                    f"{s}: rate={t.success_rate:.3f} n={t.n_samples}"
                )
            neg_msg = f"[Negative] {'; '.join(neg_stats)}"
            logger.info(neg_msg)
            print(neg_msg)

    # ---- Sampler interface ----

    def __iter__(self):
        """Yield indices in batch-aligned blocks: negative + positive per batch."""
        batch_size = self._batch_size
        neg_single_pool = self.source_indices.get("negative_single_table", [])
        neg_multi_pool = self.source_indices.get("negative_multi_table", [])
        neg_single_k = self.neg_single_per_batch
        neg_multi_k = self.neg_multi_per_batch
        pos_k = batch_size - neg_single_k - neg_multi_k

        # Positive sources and weights
        valid_pos = [(s, self._weights.get(s, 0)) for s in self.positive_sources
                     if self.source_indices.get(s)]
        if not valid_pos:
            # Fallback: yield all indices shuffled
            all_indices = list(range(len(self.dataset)))
            self.rng.shuffle(all_indices)
            yield from all_indices
            return

        pos_sources, pos_weights = zip(*valid_pos)
        w_sum = sum(pos_weights)
        pos_weights = [w / w_sum for w in pos_weights]

        # Number of full batches per epoch (based on positive sample count)
        positive_total = sum(len(self.source_indices[s]) for s in self.positive_sources)
        num_batches = max(1, positive_total // batch_size)

        for _ in range(num_batches):
            batch_indices = []

            # 1. Inject negative samples (with replacement from the pool)
            if neg_single_k > 0 and neg_single_pool:
                batch_indices.extend(self.rng.choices(neg_single_pool, k=neg_single_k))
            if neg_multi_k > 0 and neg_multi_pool:
                batch_indices.extend(self.rng.choices(neg_multi_pool, k=neg_multi_k))

            # 2. Fill the rest with curriculum-sampled positive samples
            for _ in range(pos_k):
                source = self.rng.choices(pos_sources, weights=pos_weights, k=1)[0]
                idx = self.rng.choice(self.source_indices[source])
                batch_indices.append(idx)

            # 3. Shuffle within batch to avoid positional bias
            self.rng.shuffle(batch_indices)
            yield from batch_indices

    def __len__(self):
        """Epoch length based on positive samples only (negative injected per batch)."""
        batch_size = self._batch_size
        positive_total = sum(len(self.source_indices[s]) for s in self.positive_sources)
        num_batches = max(1, positive_total // batch_size)
        return num_batches * batch_size
