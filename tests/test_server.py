"""Tests for MCP server tool functions.

Calls server functions directly (no MCP transport), asserts on response status.
Uses random embeddings to avoid downloading sentence-transformers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import prometheus.server as server
from prometheus.core.config import PrometheusConfig
from prometheus.core.lru import LRUCache
from prometheus.data.splits import DataSplit
from prometheus.data.taxonomy import TaxonomyInfo
from prometheus.inference.novelty import NoveltyDetector
from prometheus.storage.model_store import save_model
from prometheus.training.trainer import TrainingConfig
from prometheus.training.trainer import train_model as run_training

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "naics_test.csv"
TEXT_COL = "description"
LEVEL_COLS = ["Sector", "Subsector", "IndustryGroup", "NAICSIndustry"]
EMBED_DIM = 32


def _make_random_embeddings(texts, model_name=None, show_progress=False):
    """Deterministic random embeddings keyed by list length for reproducibility."""
    rng = np.random.default_rng(len(texts))
    return rng.standard_normal((len(texts), EMBED_DIM)).astype(np.float32)


@pytest.fixture(autouse=True)
def tmp_config(tmp_path, monkeypatch):
    """Replace server config with a temp directory and mock embeddings."""
    cfg = PrometheusConfig(base_dir=tmp_path / ".prometheus")
    cfg.ensure_dirs()
    monkeypatch.setattr(server, "_config", cfg)
    monkeypatch.setattr(server, "_predictor_cache", LRUCache(maxsize=8))
    monkeypatch.setattr(server, "_code_mapping_cache", LRUCache(maxsize=16))
    monkeypatch.setattr("prometheus.server.tools.inference.encode_texts", _make_random_embeddings)
    monkeypatch.setattr("prometheus.server.tools.evaluation.encode_texts", _make_random_embeddings)
    monkeypatch.setattr("prometheus.data.loader.encode_texts", _make_random_embeddings)
    return cfg


@pytest.fixture(scope="module")
def _trained_artifacts():
    """Train a model once (module-scoped) and return reusable artifacts."""
    df = pd.read_csv(FIXTURE_PATH).dropna(subset=[TEXT_COL] + LEVEL_COLS).reset_index(drop=True)

    level_classes = [sorted(df[col].unique().tolist()) for col in LEVEL_COLS]
    taxonomy = TaxonomyInfo.build(
        name="srv_test",
        text_column=TEXT_COL,
        level_columns=LEVEL_COLS,
        level_classes=level_classes,
        embedding_model="mock",
        total_samples=len(df),
    )

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((len(df), EMBED_DIM)).astype(np.float32)

    labels_by_depth = {}
    for depth, level in enumerate(taxonomy.levels):
        labels_by_depth[depth] = np.array([level.encode(v) for v in df[level.name]])

    n = len(df)
    idx = rng.permutation(n)
    t_end, v_end = int(n * 0.8), int(n * 0.9)

    def _split(i):
        return DataSplit(
            embeddings=embeddings[i],
            labels_by_depth={d: lbls[i] for d, lbls in labels_by_depth.items()},
            texts=[df[TEXT_COL].iloc[j] for j in i],
        )

    train, val = _split(idx[:t_end]), _split(idx[t_end:v_end])

    config = TrainingConfig(epochs=3, batch_size=64, hidden_dims=[64, 32], dropout=0.2)
    model, result, noise = run_training(
        taxonomy=taxonomy,
        train_data=train,
        val_data=val,
        embedding_dim=EMBED_DIM,
        config=config,
    )

    novelty = NoveltyDetector()
    novelty.fit(train.embeddings, train.leaf_labels, taxonomy.leaf_level.classes)

    return model, taxonomy, result, config, train, novelty


@pytest.fixture()
def trained_model(_trained_artifacts, tmp_config):
    """Save the pre-trained model into the temp config dir and return its name."""
    model, taxonomy, result, config, train, novelty = _trained_artifacts
    name = "test_model"
    model_path = tmp_config.model_path(name)

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
    return name


@pytest.fixture()
def sample_csv(tmp_path):
    """Write a small test CSV and return its path."""
    df = pd.read_csv(FIXTURE_PATH).head(50)
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


# ─── Training & Data ───


class TestAnalyzeTrainingData:
    def test_valid_csv(self):
        result = server.analyze_training_data(str(FIXTURE_PATH), TEXT_COL, LEVEL_COLS)
        assert result["status"] == "success"
        assert "level_stats" in result

    def test_missing_column(self):
        result = server.analyze_training_data(str(FIXTURE_PATH), "nonexistent", LEVEL_COLS)
        assert result["status"] == "error"


class TestTrainModel:
    def test_full_train(self, sample_csv):
        result = server.train_model(
            file_path=sample_csv,
            text_column=TEXT_COL,
            level_columns=LEVEL_COLS,
            model_name="train_test",
            config=TrainingConfig(epochs=2, hidden_dims=[32, 16]),
        )
        assert result["status"] == "success"
        assert "metrics" in result
        assert "learning_insights" in result

    def test_missing_file(self):
        result = server.train_model(
            file_path="/nonexistent.csv",
            text_column=TEXT_COL,
            level_columns=LEVEL_COLS,
            model_name="bad_train",
        )
        assert result["status"] == "error"

    def test_resume_training(self, trained_model, sample_csv):
        result = server.resume_training(
            model_name=trained_model,
            file_path=sample_csv,
            additional_epochs=2,
        )
        assert result["status"] == "success"
        assert "metrics" in result


# ─── Inference ───


class TestPredict:
    def test_predict(self, trained_model):
        result = server.predict(model_name=trained_model, text="Soybean Farming")
        assert result["status"] == "success"
        assert "path" in result
        assert "routing" in result

    def test_predict_greedy(self, trained_model):
        result = server.predict(model_name=trained_model, text="Soybean Farming", beam_width=1)
        assert result["status"] == "success"
        assert len(result["path"]) == 4

    def test_predict_model_not_found(self):
        result = server.predict(model_name="nonexistent", text="test")
        assert result["status"] == "error"


class TestPredictBatch:
    def test_predict_batch(self, trained_model, sample_csv):
        result = server.predict_batch(
            model_name=trained_model,
            file_path=sample_csv,
            text_column=TEXT_COL,
        )
        assert result["status"] == "success"
        assert "routing_distribution" in result

    def test_predict_batch_missing_column(self, trained_model, sample_csv):
        result = server.predict_batch(
            model_name=trained_model,
            file_path=sample_csv,
            text_column="nonexistent",
        )
        assert result["status"] == "validation_error"


class TestClassifyWithContext:
    def test_classify_with_context(self, trained_model):
        result = server.classify_with_context(
            model_name=trained_model,
            text="Soybean Farming",
            known_levels={"Sector": "Agriculture, Forestry, Fishing and Hunting"},
        )
        assert result["status"] == "success"
        assert result["path"][0] == "Agriculture, Forestry, Fishing and Hunting"


# ─── Model Management ───


class TestListModels:
    def test_empty(self):
        result = server.list_models()
        assert result["status"] == "success"
        assert result["count"] == 0

    def test_with_model(self, trained_model):
        result = server.list_models()
        assert result["status"] == "success"
        assert result["count"] == 1


class TestDescribeModel:
    def test_existing(self, trained_model):
        result = server.describe_model(trained_model)
        assert result["status"] == "success"
        assert "manifest" in result
        assert "model_health" in result

    def test_not_found(self):
        result = server.describe_model("nonexistent")
        assert result["status"] == "validation_error"


class TestDeleteModel:
    def test_existing(self, trained_model, tmp_config):
        model_path = tmp_config.model_path(trained_model)
        assert model_path.exists()
        result = server.delete_model(trained_model)
        assert result["status"] == "success"
        assert not model_path.exists()

    def test_not_found(self):
        result = server.delete_model("nonexistent")
        assert result["status"] == "validation_error"


class TestExportModel:
    def test_export(self, trained_model, tmp_path):
        dest = str(tmp_path / "exported_model")
        result = server.export_model(trained_model, dest)
        assert result["status"] == "success"
        assert Path(dest).exists()


# ─── Code Mapping ───


class TestBuildCodeMapping:
    def test_valid(self, trained_model, tmp_path):
        # Create a CSV with a code column
        df = pd.read_csv(FIXTURE_PATH).head(50)
        df["code"] = [f"C{i:04d}" for i in range(len(df))]
        csv_path = str(tmp_path / "codes.csv")
        df.to_csv(csv_path, index=False)

        result = server.build_code_mapping(
            model_name=trained_model,
            file_path=csv_path,
            code_column="code",
        )
        assert result["status"] == "success"
        assert result["unique_paths"] > 0

    def test_missing_column(self, trained_model, sample_csv):
        result = server.build_code_mapping(
            model_name=trained_model,
            file_path=sample_csv,
            code_column="nonexistent_code",
        )
        assert result["status"] == "validation_error"

    def test_model_not_found(self, sample_csv):
        result = server.build_code_mapping(
            model_name="nonexistent",
            file_path=sample_csv,
            code_column="code",
        )
        assert result["status"] == "validation_error"


# ─── Evaluation & Diagnostics ───


class TestEvaluateModel:
    def test_evaluate(self, trained_model, sample_csv):
        result = server.evaluate_model(
            model_name=trained_model,
            file_path=sample_csv,
            text_column=TEXT_COL,
            level_columns=LEVEL_COLS,
        )
        assert result["status"] == "success"
        assert "per_level_accuracy" in result


class TestExplainPrediction:
    def test_explain(self, trained_model):
        result = server.explain_prediction(model_name=trained_model, text="Soybean Farming")
        assert result["status"] == "success"
        assert "level_details" in result
        assert "novelty" in result


class TestConfusionMatrix:
    def test_valid_level(self, trained_model, sample_csv):
        result = server.get_confusion_matrix(
            model_name=trained_model,
            level="Sector",
            file_path=sample_csv,
            text_column=TEXT_COL,
            level_columns=LEVEL_COLS,
        )
        assert result["status"] == "success"
        assert "classification_report" in result

    def test_bad_level(self, trained_model, sample_csv):
        result = server.get_confusion_matrix(
            model_name=trained_model,
            level="NonexistentLevel",
            file_path=sample_csv,
            text_column=TEXT_COL,
            level_columns=LEVEL_COLS,
        )
        assert result["status"] == "validation_error"


class TestThresholdReport:
    def test_existing_model(self, trained_model):
        result = server.get_threshold_report(trained_model)
        assert result["status"] == "success"
        assert "per_level_thresholds" in result


class TestSetThresholds:
    def test_override(self, trained_model):
        result = server.set_thresholds(trained_model, {"0": 0.80})
        assert result["status"] == "success"
        # Verify persisted
        report = server.get_threshold_report(trained_model)
        assert report["per_level_thresholds"]["Sector"]["threshold"] == 0.80

    def test_by_name(self, trained_model):
        result = server.set_thresholds(trained_model, {"Sector": 0.85})
        assert result["status"] == "success"

    def test_invalidates_cache(self, trained_model):
        # Load predictor into cache
        server._get_predictor(trained_model)
        assert trained_model in server._predictor_cache

        server.set_thresholds(trained_model, {"0": 0.70})
        assert trained_model not in server._predictor_cache


# ─── Feedback ───


class TestSubmitFeedback:
    def test_valid(self, trained_model, tmp_config):
        result = server.submit_feedback(
            model_name=trained_model,
            text="test item",
            corrected_labels={"Sector": "Mining"},
        )
        assert result["status"] == "success"
        # Verify file written
        feedback_file = tmp_config.feedback_dir / trained_model / "feedback.jsonl"
        assert feedback_file.exists()
        line = json.loads(feedback_file.read_text().strip())
        assert line["text"] == "test item"

    def test_with_prediction_id(self, trained_model):
        result = server.submit_feedback(
            model_name=trained_model,
            text="test item",
            corrected_labels={"Sector": "Mining"},
            prediction_id="abc123",
        )
        assert result["status"] == "success"


# ─── Embedding ───


class TestListEmbeddingModels:
    def test_returns_models(self):
        result = server.list_embedding_models()
        assert result["status"] == "success"
        assert len(result["models"]) >= 3
