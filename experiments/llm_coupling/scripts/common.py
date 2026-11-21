"""Shared paths and the held-out split for the LLM coupling experiment."""

from __future__ import annotations

from pathlib import Path

from prometheus.data.loader import prepare_data
from prometheus.server import _get_predictor

EXP = Path(__file__).resolve().parents[1]
DATA = EXP / "data"
RUNS = EXP / "runs"
CSV = EXP.parents[1] / "data" / "hts_training.csv"
MODEL = "hts_viz"
LEVELS = ["Chapter", "Heading", "Subheading"]


def load():
    """The trained model plus the exact train/test splits it was trained with (random_state=42)."""
    predictor = _get_predictor(MODEL)
    _taxonomy, train, _val, test, _warnings = prepare_data(
        file_path=str(CSV),
        text_column="text",
        level_columns=LEVELS,
        model_name=MODEL,
        embedding_model=predictor.taxonomy.embedding_model,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    return predictor, train, test


def beam_paths(predictor, embedding, width: int) -> list[list[int]]:
    """Surviving beams, best first, as full index paths."""
    predictor.beam.beam_width = width
    cp = predictor.beam.predict(embedding)
    return [cp.path_indices] + [b.indices for b in cp.alternative_paths]


def leaves_under_headings(taxonomy, headings) -> set[int]:
    """Every subheading under the given headings, from the taxonomy (trained on or not)."""
    pcm = taxonomy.parent_child_map
    return set().union(*(set(pcm.get((2, h), [])) for h in headings))
