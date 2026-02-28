"""Three-tier routing: AUTO_ACCEPT / HUMAN_REVIEW / OUT_OF_TAXONOMY."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from prometheus.inference.beam_search import BeamPath, CascadePrediction
from prometheus.inference.novelty import NoveltyScore


class RoutingDecision(Enum):
    AUTO_ACCEPT = "AUTO_ACCEPT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    OUT_OF_TAXONOMY = "OUT_OF_TAXONOMY"


@dataclass
class ForkedCompletion:
    """A candidate completion below the fork point."""

    suffix: list[str]  # labels from fork_depth onward
    suffix_indices: list[int]  # indices from fork_depth onward
    suffix_score: float  # cumulative log prob of the suffix portion
    per_level_probs: list[float]  # prob at each suffix level


@dataclass
class RoutingResult:
    decision: RoutingDecision
    confident_depth: int | None  # deepest confident level depth (or None)
    confident_path: list[str]  # path up to confident_depth
    full_path: list[str]  # full predicted path regardless
    review_reason: str | None
    forked_completions: list[ForkedCompletion] | None = None


class ThresholdRouter:
    """Routes predictions based on per-level confidence thresholds."""

    def __init__(self, thresholds: dict, level_names: list[str]):
        """
        Args:
            thresholds: {depth_int: confidence_threshold, "margin": float, "z_score": float, "novelty_conf": float}
            level_names: ordered level names for display
        """
        self.thresholds = thresholds
        self.level_names = level_names
        self.margin_threshold = thresholds.get("margin", 0.15)

    def route(self, prediction: CascadePrediction, novelty: NoveltyScore) -> RoutingResult:
        if novelty.is_novel:
            return RoutingResult(
                decision=RoutingDecision.OUT_OF_TAXONOMY,
                confident_depth=None,
                confident_path=[],
                full_path=prediction.path,
                review_reason=f"Embedding distant from known classes (z={novelty.centroid_z_score:.2f})",
            )

        # Find deepest confident level
        confident_path = []
        confident_depth = None
        review_reason = None

        for lp in prediction.level_predictions:
            threshold = self.thresholds.get(lp.depth, 0.75)
            if lp.confidence >= threshold:
                confident_path.append(lp.predicted)
                confident_depth = lp.depth
            else:
                review_reason = (
                    f"{lp.name} confidence below threshold ({lp.confidence:.2f} < {threshold:.2f})"
                )
                break

        # Check margin
        for lp in prediction.level_predictions[: len(confident_path)]:
            if lp.margin < self.margin_threshold:
                idx = lp.depth
                confident_path = confident_path[:idx]
                confident_depth = idx - 1 if idx > 0 else None
                review_reason = (
                    f"{lp.name} margin too narrow ({lp.margin:.2f} < {self.margin_threshold:.2f})"
                )
                break

        # Determine decision
        num_levels = len(prediction.level_predictions)
        if confident_depth is not None and confident_depth == num_levels - 1:
            decision = RoutingDecision.AUTO_ACCEPT
        else:
            decision = RoutingDecision.HUMAN_REVIEW

        return RoutingResult(
            decision=decision,
            confident_depth=confident_depth,
            confident_path=confident_path,
            full_path=prediction.path,
            review_reason=review_reason,
        )

    def route_with_forks(
        self, prediction: CascadePrediction, novelty: NoveltyScore
    ) -> RoutingResult:
        """Route with forked completion candidates below the confidence drop."""
        result = self.route(prediction, novelty)

        if not prediction.alternative_paths:
            return result

        fork_depth = (result.confident_depth + 1) if result.confident_depth is not None else 0
        num_levels = len(prediction.level_predictions)

        # Confident through leaf — AUTO_ACCEPT, no fork needed
        if fork_depth >= num_levels:
            return result

        # Reconstruct best path as BeamPath for uniform processing
        best = BeamPath(
            indices=prediction.path_indices,
            labels=prediction.path,
            probs=[lp.confidence for lp in prediction.level_predictions],
            log_score=prediction.path_score,
        )

        all_beams = [best] + prediction.alternative_paths

        if fork_depth == 0:
            # No confident prefix — group by L1 class, best path per branch
            seen: dict[int, ForkedCompletion] = {}
            for bp in all_beams:
                branch_key = bp.indices[0]
                suffix_probs = bp.probs
                suffix_score = sum(math.log(p + 1e-10) for p in suffix_probs)
                if branch_key not in seen or suffix_score > seen[branch_key].suffix_score:
                    seen[branch_key] = ForkedCompletion(
                        suffix=bp.labels,
                        suffix_indices=list(bp.indices),
                        suffix_score=round(suffix_score, 4),
                        per_level_probs=[round(p, 4) for p in suffix_probs],
                    )
        else:
            # Confident prefix exists — collect paths sharing it
            confident_prefix = prediction.path_indices[:fork_depth]
            seen: dict[tuple[int, ...], ForkedCompletion] = {}
            for bp in all_beams:
                if bp.indices[:fork_depth] != confident_prefix:
                    continue
                suffix_key = tuple(bp.indices[fork_depth:])
                suffix_probs = bp.probs[fork_depth:]
                suffix_score = sum(math.log(p + 1e-10) for p in suffix_probs)
                if suffix_key not in seen or suffix_score > seen[suffix_key].suffix_score:
                    seen[suffix_key] = ForkedCompletion(
                        suffix=bp.labels[fork_depth:],
                        suffix_indices=list(bp.indices[fork_depth:]),
                        suffix_score=round(suffix_score, 4),
                        per_level_probs=[round(p, 4) for p in suffix_probs],
                    )

        forks = sorted(seen.values(), key=lambda f: f.suffix_score, reverse=True)
        result.forked_completions = forks if forks else None

        return result
