"""N-level beam search and greedy inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from prometheus.data.taxonomy import TaxonomyInfo
from prometheus.training.cascade import HierarchicalClassifier


@dataclass
class BeamPath:
    """A single path hypothesis in beam search."""

    indices: list[int]
    labels: list[str]
    probs: list[float]
    log_score: float


@dataclass
class LevelPrediction:
    """Prediction details for a single level."""

    depth: int
    name: str
    predicted: str
    predicted_idx: int
    confidence: float
    margin: float
    top_candidates: list[dict]


@dataclass
class CascadePrediction:
    """Full prediction result."""

    path: list[str]
    path_indices: list[int]
    path_score: float
    level_predictions: list[LevelPrediction]  # ordered by depth
    alternative_paths: list[BeamPath]


class BeamSearchClassifier:
    """N-level beam search classifier."""

    def __init__(
        self,
        model: HierarchicalClassifier,
        taxonomy: TaxonomyInfo,
        device: torch.device | None = None,
        beam_width: int = 3,
        top_k_candidates: int = 3,
        parent_child_map: dict[tuple[int, int], list[int]] | None = None,
        confidence_thresholds: dict[int, float] | None = None,
        adaptive_factor: int = 2,
    ):
        self.model = model
        self.taxonomy = taxonomy
        self.device = device or torch.device("cpu")
        self.beam_width = beam_width
        self.top_k_candidates = top_k_candidates
        self.parent_child_map = parent_child_map
        self.confidence_thresholds = confidence_thresholds
        self.adaptive_factor = adaptive_factor

    def predict(self, embedding: np.ndarray) -> CascadePrediction:
        self.model.eval()
        self._diversity_depth: int | None = None

        emb = torch.tensor(embedding, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            # Level 0: init beams
            level0 = self.taxonomy.get_level(0)
            logits = self.model.get_level(0)(emb)
            probs0 = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            beams = self._init_beams(probs0, level0.classes)
            level_probs = {0: probs0}

            # Levels 1..N-1: expand beams
            for depth in range(1, self.taxonomy.num_levels):
                level = self.taxonomy.get_level(depth)
                beams, probs = self._expand_beams(emb, beams, depth, level.classes)
                level_probs[depth] = probs

        best = beams[0]
        level_predictions = self._build_level_predictions(best, level_probs)

        return CascadePrediction(
            path=best.labels,
            path_indices=best.indices,
            path_score=best.log_score,
            level_predictions=level_predictions,
            alternative_paths=beams[1:],
        )

    def predict_batch(self, embeddings: np.ndarray) -> list[CascadePrediction]:
        return [self.predict(emb) for emb in embeddings]

    def _init_beams(self, probs: np.ndarray, classes: list[str]) -> list[BeamPath]:
        effective_width = self.beam_width
        if self.confidence_thresholds and 0 in self.confidence_thresholds:
            if float(probs.max()) < self.confidence_thresholds[0]:
                effective_width = self.beam_width * self.adaptive_factor
                if self._diversity_depth is None:
                    self._diversity_depth = 0

        top_k = np.argsort(probs)[::-1][:effective_width]
        return [
            BeamPath(
                indices=[int(idx)],
                labels=[classes[idx]],
                probs=[float(probs[idx])],
                log_score=float(np.log(probs[idx] + 1e-10)),
            )
            for idx in top_k
        ]

    def _expand_beams(
        self,
        embedding: torch.Tensor,
        beams: list[BeamPath],
        depth: int,
        classes: list[str],
    ) -> tuple[list[BeamPath], np.ndarray]:
        classifier = self.model.get_level(depth)
        new_beams = []
        best_probs = None
        effective_width = self.beam_width

        for beam in beams:
            context = self._build_beam_context(beam, depth)
            logits = classifier(embedding, context)

            # Mask invalid children if parent_child_map is available
            if self.parent_child_map is not None:
                key = (depth, beam.indices[depth - 1])
                valid = self.parent_child_map.get(key)
                if valid is not None:
                    mask = torch.full_like(logits, float("-inf"))
                    mask[0, valid] = 0.0
                    logits = logits + mask

            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            if best_probs is None:
                best_probs = probs
                # Adaptive: widen if best beam's top prob is below threshold
                if self.confidence_thresholds and depth in self.confidence_thresholds:
                    if float(probs.max()) < self.confidence_thresholds[depth]:
                        effective_width = self.beam_width * self.adaptive_factor
                        if self._diversity_depth is None:
                            self._diversity_depth = depth

            top_k = np.argsort(probs)[::-1][:effective_width]
            for idx in top_k:
                new_beams.append(
                    BeamPath(
                        indices=beam.indices + [int(idx)],
                        labels=beam.labels + [classes[idx]],
                        probs=beam.probs + [float(probs[idx])],
                        log_score=beam.log_score + float(np.log(probs[idx] + 1e-10)),
                    )
                )

        new_beams.sort(key=lambda b: b.log_score, reverse=True)
        pruned = self._prune_with_diversity(new_beams, effective_width)
        return pruned, best_probs

    def _prune_with_diversity(self, beams: list[BeamPath], width: int) -> list[BeamPath]:
        """Prune beams, guaranteeing representation from each branch at the uncertain depth."""
        if self._diversity_depth is None or len(beams) <= width:
            return beams[:width]

        dd = self._diversity_depth
        # beams are pre-sorted by log_score descending

        # Phase 1: best beam from each unique class at diversity depth
        seen: dict[int, BeamPath] = {}
        remainder: list[BeamPath] = []
        for beam in beams:
            cls = beam.indices[dd]
            if cls not in seen:
                seen[cls] = beam
            else:
                remainder.append(beam)

        representatives = list(seen.values())
        if len(representatives) >= width:
            return representatives[:width]

        # Phase 2: fill remaining slots by global score
        result = list(representatives)
        result.extend(remainder[: width - len(result)])
        result.sort(key=lambda b: b.log_score, reverse=True)
        return result

    def _build_beam_context(self, beam: BeamPath, depth: int) -> torch.Tensor:
        context_parts = []
        for d in range(depth):
            num_cls = self.taxonomy.get_level(d).num_classes
            one_hot = torch.zeros(1, num_cls, device=self.device)
            one_hot[0, beam.indices[d]] = 1.0
            context_parts.append(one_hot)
        return torch.cat(context_parts, dim=-1)

    def _build_level_predictions(
        self, beam: BeamPath, level_probs: dict[int, np.ndarray]
    ) -> list[LevelPrediction]:
        predictions = []
        for depth in range(self.taxonomy.num_levels):
            level = self.taxonomy.get_level(depth)
            probs = level_probs[depth]
            sorted_idx = np.argsort(probs)[::-1]

            top_candidates = [
                {"label": level.classes[idx], "confidence": float(probs[idx])}
                for idx in sorted_idx[: self.top_k_candidates]
            ]

            margin = (
                float(probs[sorted_idx[0]] - probs[sorted_idx[1]]) if len(sorted_idx) >= 2 else 1.0
            )

            predictions.append(
                LevelPrediction(
                    depth=depth,
                    name=level.name,
                    predicted=beam.labels[depth],
                    predicted_idx=beam.indices[depth],
                    confidence=beam.probs[depth],
                    margin=margin,
                    top_candidates=top_candidates,
                )
            )
        return predictions


class GreedyClassifier:
    """N-level greedy classifier."""

    def __init__(
        self,
        model: HierarchicalClassifier,
        taxonomy: TaxonomyInfo,
        device: torch.device | None = None,
        top_k_candidates: int = 3,
        parent_child_map: dict[tuple[int, int], list[int]] | None = None,
    ):
        self.model = model
        self.taxonomy = taxonomy
        self.device = device or torch.device("cpu")
        self.top_k_candidates = top_k_candidates
        self.parent_child_map = parent_child_map

    def predict(self, embedding: np.ndarray) -> CascadePrediction:
        self.model.eval()

        emb = torch.tensor(embedding, dtype=torch.float32, device=self.device).unsqueeze(0)

        path_labels, path_indices, path_probs = [], [], []
        path_log_score = 0.0
        level_predictions = []
        context_parts = []

        with torch.no_grad():
            for depth in range(self.taxonomy.num_levels):
                level = self.taxonomy.get_level(depth)
                classifier = self.model.get_level(depth)

                context = torch.cat(context_parts, dim=-1) if context_parts else None
                logits = classifier(emb, context)

                # Mask invalid children if parent_child_map is available
                if depth > 0 and self.parent_child_map is not None:
                    key = (depth, path_indices[-1])
                    valid = self.parent_child_map.get(key)
                    if valid is not None:
                        mask = torch.full_like(logits, float("-inf"))
                        mask[0, valid] = 0.0
                        logits = logits + mask

                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                sorted_idx = np.argsort(probs)[::-1]
                pred_idx = int(sorted_idx[0])
                pred_prob = float(probs[pred_idx])

                path_labels.append(level.classes[pred_idx])
                path_indices.append(pred_idx)
                path_probs.append(pred_prob)
                path_log_score += float(np.log(pred_prob + 1e-10))

                top_candidates = [
                    {"label": level.classes[idx], "confidence": float(probs[idx])}
                    for idx in sorted_idx[: self.top_k_candidates]
                ]

                margin = (
                    float(probs[sorted_idx[0]] - probs[sorted_idx[1]])
                    if len(sorted_idx) >= 2
                    else 1.0
                )

                level_predictions.append(
                    LevelPrediction(
                        depth=depth,
                        name=level.name,
                        predicted=level.classes[pred_idx],
                        predicted_idx=pred_idx,
                        confidence=pred_prob,
                        margin=margin,
                        top_candidates=top_candidates,
                    )
                )

                one_hot = torch.zeros(1, level.num_classes, device=self.device)
                one_hot[0, pred_idx] = 1.0
                context_parts.append(one_hot)

        return CascadePrediction(
            path=path_labels,
            path_indices=path_indices,
            path_score=path_log_score,
            level_predictions=level_predictions,
            alternative_paths=[],
        )

    def predict_batch(self, embeddings: np.ndarray) -> list[CascadePrediction]:
        return [self.predict(emb) for emb in embeddings]
