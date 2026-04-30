"""CascadePredictor: orchestrates embed -> classify -> route."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import numpy as np
import torch

from prometheus.data.taxonomy import TaxonomyInfo
from prometheus.inference.beam_search import (
    BeamSearchClassifier,
    GreedyClassifier,
)
from prometheus.inference.novelty import NoveltyDetector
from prometheus.inference.router import ThresholdRouter
from prometheus.training.cascade import HierarchicalClassifier

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    prediction_id: str
    input_text: str
    predictions: list[dict]
    path: list[str]
    path_score: float
    novelty: dict
    routing: dict
    forked_completions: list[dict] | None = None


class CascadePredictor:
    """Main prediction interface."""

    def __init__(
        self,
        model: HierarchicalClassifier,
        taxonomy: TaxonomyInfo,
        novelty_detector: NoveltyDetector,
        router: ThresholdRouter,
        device: torch.device | None = None,
        beam_width: int = 3,
        parent_child_map: dict[tuple[int, int], list[int]] | None = None,
        adaptive_factor: int = 2,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.taxonomy = taxonomy
        self.novelty_detector = novelty_detector
        self.router = router

        self.greedy = GreedyClassifier(
            model=model,
            taxonomy=taxonomy,
            device=self.device,
            parent_child_map=parent_child_map,
        )
        self.beam = BeamSearchClassifier(
            model=model,
            taxonomy=taxonomy,
            device=self.device,
            beam_width=beam_width,
            parent_child_map=parent_child_map,
            confidence_thresholds=router.config.level_thresholds,
            adaptive_factor=adaptive_factor,
        )

    def predict(
        self,
        embedding: np.ndarray,
        text: str = "",
        use_beam_search: bool = False,
        beam_width: int | None = None,
    ) -> PredictionResult:
        prediction_id = str(uuid.uuid4())[:12]

        if beam_width is not None:
            self.beam.beam_width = beam_width

        classifier = self.beam if use_beam_search else self.greedy
        cascade_result = classifier.predict(embedding)

        # Novelty on leaf prediction
        leaf_pred = cascade_result.level_predictions[-1]
        min_conf = min(lp.confidence for lp in cascade_result.level_predictions)
        novelty = self.novelty_detector.score(embedding, leaf_pred.predicted, min_conf)

        # Route (with fork detection for beam search)
        routing = self.router.route_with_forks(cascade_result, novelty)

        # Format per-level predictions
        predictions = []
        for lp in cascade_result.level_predictions:
            predictions.append(
                {
                    "depth": lp.depth,
                    "level": lp.name,
                    "predicted": lp.predicted,
                    "confidence": round(lp.confidence, 4),
                    "margin": round(lp.margin, 4),
                    "top_candidates": [
                        {"label": c["label"], "confidence": round(c["confidence"], 4)}
                        for c in lp.top_candidates
                    ],
                }
            )

        # Format forked completions
        forked = None
        if routing.forked_completions:
            forked = [
                {
                    "suffix": fc.suffix,
                    "suffix_indices": fc.suffix_indices,
                    "score": fc.suffix_score,
                    "per_level_probs": fc.per_level_probs,
                }
                for fc in routing.forked_completions
            ]

        logger.info(
            "prediction=%s routing=%s novel=%s path=%s",
            prediction_id,
            routing.decision.value,
            novelty.is_novel,
            cascade_result.path,
        )

        return PredictionResult(
            prediction_id=prediction_id,
            input_text=text,
            predictions=predictions,
            path=cascade_result.path,
            path_score=round(cascade_result.path_score, 4),
            novelty={
                "centroid_z_score": round(novelty.centroid_z_score, 4),
                "knn_distance": round(novelty.knn_distance, 4),
                "is_novel": novelty.is_novel,
            },
            routing={
                "decision": routing.decision.value,
                "confident_depth": routing.confident_depth,
                "confident_path": routing.confident_path,
                "review_reason": routing.review_reason,
            },
            forked_completions=forked,
        )

    def predict_batch(
        self,
        embeddings: np.ndarray,
        texts: list[str] | None = None,
        use_beam_search: bool = False,
        beam_width: int | None = None,
    ) -> list[PredictionResult]:
        texts = texts or [""] * len(embeddings)
        return [
            self.predict(emb, text=t, use_beam_search=use_beam_search, beam_width=beam_width)
            for emb, t in zip(embeddings, texts)
        ]
