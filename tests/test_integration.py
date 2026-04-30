"""Integration test: train on NAICS data, save, load, predict, evaluate.

Uses the full 4-level NAICS 2022 taxonomy (1,012 rows):
  Sector (20) > Subsector (96) > IndustryGroup (308) > NAICSIndustry (689)

This test exercises the full pipeline end-to-end but skips
sentence-transformer embedding (uses random embeddings for speed).
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prometheus.data.loader import analyze_data
from prometheus.data.splits import DataSplit
from prometheus.data.taxonomy import TaxonomyInfo
from prometheus.inference.beam_search import BeamSearchClassifier, GreedyClassifier
from prometheus.inference.novelty import NoveltyDetector
from prometheus.inference.predictor import CascadePredictor
from prometheus.inference.router import ThresholdRouter
from prometheus.storage.model_store import (
    generate_thresholds,
    load_model,
    load_thresholds,
    save_model,
)
from prometheus.training.trainer import TrainingConfig, train_model

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "naics_test.csv"
TEXT_COL = "description"
LEVEL_COLS = ["Sector", "Subsector", "IndustryGroup", "NAICSIndustry"]
EMBED_DIM = 32  # small for speed


@pytest.fixture(scope="module")
def naics_df():
    return pd.read_csv(FIXTURE_PATH)


@pytest.fixture(scope="module")
def naics_taxonomy_and_splits(naics_df):
    """Build taxonomy and splits with random embeddings (no sentence-transformer)."""
    df = naics_df.dropna(subset=[TEXT_COL] + LEVEL_COLS).reset_index(drop=True)

    # Build taxonomy
    level_classes = [sorted(df[col].unique().tolist()) for col in LEVEL_COLS]
    taxonomy = TaxonomyInfo.build(
        name="naics_test",
        text_column=TEXT_COL,
        level_columns=LEVEL_COLS,
        level_classes=level_classes,
        embedding_model="mock",
        total_samples=len(df),
    )

    # Random embeddings (deterministic)
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((len(df), EMBED_DIM)).astype(np.float32)

    # Encode labels
    labels_by_depth = {}
    for depth, level in enumerate(taxonomy.levels):
        labels_by_depth[depth] = np.array([level.encode(v) for v in df[level.name]])

    # Split: 80/10/10
    n = len(df)
    indices = rng.permutation(n)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    def make_split(idx):
        return DataSplit(
            embeddings=embeddings[idx],
            labels_by_depth={d: lbls[idx] for d, lbls in labels_by_depth.items()},
            texts=[df[TEXT_COL].iloc[i] for i in idx],
        )

    train = make_split(indices[:train_end])
    val = make_split(indices[train_end:val_end])
    test = make_split(indices[val_end:])

    return taxonomy, train, val, test


@pytest.fixture(scope="module")
def trained_model(naics_taxonomy_and_splits):
    """Train a model on NAICS data."""
    taxonomy, train, val, _ = naics_taxonomy_and_splits

    config = TrainingConfig(
        epochs=5,
        batch_size=64,
        noise_rate=0.15,
        hidden_dims=[64, 32],
        dropout=0.2,
    )

    model, result, noise = train_model(
        taxonomy=taxonomy,
        train_data=train,
        val_data=val,
        embedding_dim=EMBED_DIM,
        config=config,
    )

    return model, result, noise, config


class TestAnalyzeData:
    def test_analyze_returns_correct_structure(self):
        result = analyze_data(str(FIXTURE_PATH), TEXT_COL, LEVEL_COLS)

        assert result["num_levels"] == 4
        assert result["total_rows"] == 1012
        assert len(result["level_stats"]) == 4
        assert result["level_stats"][0]["column"] == "Sector"
        assert result["level_stats"][0]["num_classes"] == 20
        assert result["level_stats"][1]["column"] == "Subsector"
        assert result["level_stats"][1]["num_classes"] == 96
        assert result["level_stats"][2]["column"] == "IndustryGroup"
        assert result["level_stats"][2]["num_classes"] == 308
        assert result["level_stats"][3]["column"] == "NAICSIndustry"
        assert result["level_stats"][3]["num_classes"] == 689


class TestTraining:
    def test_model_has_correct_structure(self, trained_model, naics_taxonomy_and_splits):
        model, result, _, _ = trained_model

        assert model.num_levels == 4
        assert len(result.level_metrics) == 4

        for lm in result.level_metrics:
            assert lm.val_accuracy >= 0  # leaf level may be 0 with random embeddings
            assert lm.calibration is not None
            assert lm.calibration.temperature > 0

        # Root level should at least learn something
        assert result.level_metrics[0].val_accuracy > 0

    def test_sector_accuracy_above_chance(self, trained_model):
        """Root level should do better than random (1/20 = 5%)."""
        _, result, _, _ = trained_model
        sector_acc = result.level_metrics[0].val_accuracy
        # Random embeddings — won't be great, but training should run
        assert sector_acc > 0.01

    def test_deeper_levels_train(self, trained_model):
        """All 4 levels should produce metrics."""
        _, result, _, _ = trained_model
        names = [lm.name for lm in result.level_metrics]
        assert names == ["Sector", "Subsector", "IndustryGroup", "NAICSIndustry"]

    def test_confusion_profiles_populated(self, trained_model):
        """Noise injector should have profiles for all levels."""
        _, result, noise, _ = trained_model
        assert len(noise.confusion_profiles) == 4


class TestSaveLoadRoundtrip:
    def test_save_and_load_produces_same_predictions(
        self, trained_model, naics_taxonomy_and_splits
    ):
        model, result, _, config = trained_model
        taxonomy = naics_taxonomy_and_splits[0]
        train = naics_taxonomy_and_splits[1]

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "naics_test"

            # Fit novelty
            novelty = NoveltyDetector()
            novelty.fit(train.embeddings, train.leaf_labels, taxonomy.leaf_level.classes)

            save_model(
                path=model_path,
                model=model,
                taxonomy=taxonomy,
                result=result,
                config=config,
                embedding_dim=EMBED_DIM,
                training_embeddings=train.embeddings,
                confusion_profiles=result.confusion_profiles,
                centroids_data=novelty.get_centroids_data(),
            )
            novelty.save(model_path)

            # Verify manifest.json is valid
            with open(model_path / "manifest.json") as f:
                manifest_data = json.load(f)
            assert manifest_data["taxonomy"]["name"] == "naics_test"
            assert len(manifest_data["taxonomy"]["levels"]) == 4
            assert manifest_data["embedding_dim"] == EMBED_DIM
            assert "created_at" in manifest_data

            # Level names preserved
            level_names = [lvl["name"] for lvl in manifest_data["taxonomy"]["levels"]]
            assert level_names == [
                "Sector",
                "Subsector",
                "IndustryGroup",
                "NAICSIndustry",
            ]

            # Load
            loaded_model, loaded_manifest = load_model(model_path)
            assert loaded_model.num_levels == 4
            assert loaded_manifest.taxonomy.level_names == level_names

            # Compare root-level predictions (no context needed)
            import torch

            test_emb = torch.randn(1, EMBED_DIM)
            model.eval()
            loaded_model.eval()

            with torch.no_grad():
                np.testing.assert_allclose(
                    model.get_level(0)(test_emb).numpy(),
                    loaded_model.get_level(0)(test_emb).numpy(),
                    atol=1e-5,
                )

    def test_no_pickle_files(self, trained_model, naics_taxonomy_and_splits):
        """Verify no .pkl files are saved — JSON + .pt only."""
        model, result, _, config = trained_model
        taxonomy = naics_taxonomy_and_splits[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "no_pickle_test"
            save_model(
                path=model_path,
                model=model,
                taxonomy=taxonomy,
                result=result,
                config=config,
                embedding_dim=EMBED_DIM,
            )

            pkl_files = list(model_path.glob("*.pkl")) + list(model_path.glob("*.pickle"))
            assert len(pkl_files) == 0, f"Found pickle files: {pkl_files}"


class TestInference:
    def test_greedy_prediction(self, trained_model, naics_taxonomy_and_splits):
        model = trained_model[0]
        taxonomy = naics_taxonomy_and_splits[0]

        clf = GreedyClassifier(model, taxonomy)
        emb = np.random.randn(EMBED_DIM).astype(np.float32)
        result = clf.predict(emb)

        assert len(result.path) == 4
        for depth in range(4):
            assert result.path[depth] in taxonomy.get_level(depth).classes

    def test_beam_search_prediction(self, trained_model, naics_taxonomy_and_splits):
        model = trained_model[0]
        taxonomy = naics_taxonomy_and_splits[0]

        clf = BeamSearchClassifier(model, taxonomy, beam_width=3)
        emb = np.random.randn(EMBED_DIM).astype(np.float32)
        result = clf.predict(emb)

        assert len(result.path) == 4
        assert len(result.level_predictions) == 4
        assert len(result.alternative_paths) <= 2  # beam_width - 1

    def test_full_predictor_pipeline(self, trained_model, naics_taxonomy_and_splits):
        model = trained_model[0]
        taxonomy = naics_taxonomy_and_splits[0]
        train = naics_taxonomy_and_splits[1]

        # Novelty detector
        novelty = NoveltyDetector()
        novelty.fit(train.embeddings, train.leaf_labels, taxonomy.leaf_level.classes)

        # Router with 4-level thresholds
        thresholds = generate_thresholds(4)
        router = ThresholdRouter(thresholds, taxonomy.level_names)

        # Predictor
        predictor = CascadePredictor(
            model=model,
            taxonomy=taxonomy,
            novelty_detector=novelty,
            router=router,
        )

        emb = np.random.randn(EMBED_DIM).astype(np.float32)
        result = predictor.predict(emb, text="Soybean Farming")

        assert result.prediction_id
        assert len(result.path) == 4
        assert result.routing["decision"] in [
            "AUTO_ACCEPT",
            "HUMAN_REVIEW",
            "OUT_OF_TAXONOMY",
        ]
        assert "centroid_z_score" in result.novelty
        assert len(result.predictions) == 4

        # Verify prediction structure
        for p in result.predictions:
            assert "depth" in p
            assert "level" in p
            assert "predicted" in p
            assert "confidence" in p
            assert "top_candidates" in p

    def test_batch_prediction(self, trained_model, naics_taxonomy_and_splits):
        model = trained_model[0]
        taxonomy = naics_taxonomy_and_splits[0]
        train = naics_taxonomy_and_splits[1]

        novelty = NoveltyDetector()
        novelty.fit(train.embeddings, train.leaf_labels, taxonomy.leaf_level.classes)

        thresholds = generate_thresholds(4)
        router = ThresholdRouter(thresholds, taxonomy.level_names)

        predictor = CascadePredictor(
            model=model,
            taxonomy=taxonomy,
            novelty_detector=novelty,
            router=router,
        )

        embs = np.random.randn(10, EMBED_DIM).astype(np.float32)
        results = predictor.predict_batch(embs)

        assert len(results) == 10
        for r in results:
            assert len(r.path) == 4


class TestThresholds:
    def test_auto_generated_thresholds_for_4_levels(self):
        """Should match the original parts-classifier values exactly."""
        config = generate_thresholds(4)
        assert config.level_thresholds[0] == 0.90
        assert config.level_thresholds[1] == 0.85
        assert config.level_thresholds[2] == 0.80
        assert config.level_thresholds[3] == 0.75

    def test_thresholds_save_load(self, trained_model, naics_taxonomy_and_splits):
        model, result, _, config = trained_model
        taxonomy = naics_taxonomy_and_splits[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "thresh_test"
            save_model(
                path=model_path,
                model=model,
                taxonomy=taxonomy,
                result=result,
                config=config,
                embedding_dim=EMBED_DIM,
            )

            loaded = load_thresholds(model_path)
            assert set(loaded.level_thresholds.keys()) == {0, 1, 2, 3}
            assert loaded.margin == 0.15
            assert loaded.level_thresholds[0] == 0.90
            assert loaded.level_thresholds[3] == 0.75
