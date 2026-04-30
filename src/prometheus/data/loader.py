"""CSV/JSON data loading, taxonomy extraction, embedding, and splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from prometheus.core.embedding import encode_texts
from prometheus.data.splits import DataSplit
from prometheus.data.taxonomy import TaxonomyInfo


def load_dataframe(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV or JSON file into a DataFrame."""
    path = Path(file_path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def validate_columns(
    df: pd.DataFrame,
    text_column: str,
    level_columns: list[str],
) -> list[str]:
    """Validate that required columns exist. Returns list of warnings."""
    warnings = []
    missing = []

    if text_column not in df.columns:
        missing.append(text_column)
    for col in level_columns:
        if col not in df.columns:
            missing.append(col)

    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    # Check for missing values
    all_cols = [text_column] + level_columns
    null_counts = df[all_cols].isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            warnings.append(f"Column '{col}' has {count} missing values")

    return warnings


def check_hierarchy_consistency(
    df: pd.DataFrame,
    level_columns: list[str],
) -> list[str]:
    """Check that child labels map to consistent parents.

    Returns list of warnings for inconsistencies.
    """
    warnings = []
    for i in range(1, len(level_columns)):
        child_col = level_columns[i]
        parent_col = level_columns[i - 1]

        # Group child values by parent
        child_parents = df.groupby(child_col)[parent_col].nunique()
        multi_parent = child_parents[child_parents > 1]

        if len(multi_parent) > 0:
            for child_val, n_parents in multi_parent.items():
                parents = df[df[child_col] == child_val][parent_col].unique()
                warnings.append(
                    f"'{child_val}' at level '{child_col}' maps to {n_parents} "
                    f"parents at '{parent_col}': {list(parents)}"
                )
    return warnings


def extract_taxonomy(
    df: pd.DataFrame,
    model_name: str,
    text_column: str,
    level_columns: list[str],
    embedding_model: str,
) -> TaxonomyInfo:
    """Extract taxonomy structure from a DataFrame."""
    level_classes = []
    for col in level_columns:
        classes = sorted(df[col].dropna().unique().tolist())
        level_classes.append(classes)

    return TaxonomyInfo.build(
        name=model_name,
        text_column=text_column,
        level_columns=level_columns,
        level_classes=level_classes,
        embedding_model=embedding_model,
        total_samples=len(df),
    )


def prepare_data(
    file_path: str | Path,
    text_column: str,
    level_columns: list[str],
    model_name: str,
    embedding_model: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_state: int = 42,
    show_progress: bool = True,
) -> tuple[TaxonomyInfo, DataSplit, DataSplit, DataSplit, list[str]]:
    """Full data preparation pipeline: load, validate, embed, split.

    Returns:
        (taxonomy, train_split, val_split, test_split, warnings)
    """
    df = load_dataframe(file_path)
    warnings = validate_columns(df, text_column, level_columns)

    # Drop rows with missing values in required columns
    all_cols = [text_column] + level_columns
    clean_df = df.dropna(subset=all_cols).reset_index(drop=True)
    if len(clean_df) < len(df):
        dropped = len(df) - len(clean_df)
        warnings.append(f"Dropped {dropped} rows with missing values")

    warnings.extend(check_hierarchy_consistency(clean_df, level_columns))

    # Extract taxonomy
    taxonomy = extract_taxonomy(clean_df, model_name, text_column, level_columns, embedding_model)

    # Build parent→valid_children map for hierarchical masking
    if len(level_columns) > 1:
        pcm: dict[tuple[int, int], list[int]] = {}
        for depth in range(1, len(level_columns)):
            parent_col = level_columns[depth - 1]
            child_col = level_columns[depth]
            parent_level = taxonomy.get_level(depth - 1)
            child_level = taxonomy.get_level(depth)
            for parent_cls in parent_level.classes:
                child_vals = clean_df.loc[clean_df[parent_col] == parent_cls, child_col].unique()
                child_indices = sorted(child_level.encode(c) for c in child_vals)
                parent_idx = parent_level.encode(parent_cls)
                pcm[(depth, parent_idx)] = child_indices
        taxonomy.parent_child_map = pcm

    # Encode texts
    texts = clean_df[text_column].tolist()
    embeddings = encode_texts(texts, embedding_model, show_progress=show_progress)

    # Encode labels by depth
    labels_by_depth = {}
    for depth, level in enumerate(taxonomy.levels):
        labels_by_depth[depth] = np.array([level.encode(val) for val in clean_df[level.name]])

    # Split — handle rare classes that can't be stratified
    indices = np.arange(len(clean_df))
    root_labels = labels_by_depth[0]

    # Find classes too rare to stratify (need ≥2 samples in val+test,
    # which means ≥ ceil(2 / val_test_ratio) total samples)
    val_test_ratio = val_ratio + test_ratio
    unique_labels, label_counts = np.unique(root_labels, return_counts=True)
    min_samples_for_stratify = max(2, int(np.ceil(2 / val_test_ratio)))
    rare_labels = set(unique_labels[label_counts < min_samples_for_stratify])

    if rare_labels:
        # Rare classes go straight into training; stratify the rest
        rare_mask = np.isin(root_labels, list(rare_labels))
        rare_idx = indices[rare_mask]
        stratifiable_idx = indices[~rare_mask]
        stratifiable_labels = root_labels[stratifiable_idx]

        strat_train_idx, val_test_idx = train_test_split(
            stratifiable_idx,
            test_size=val_test_ratio,
            random_state=random_state,
            stratify=stratifiable_labels,
        )
        train_idx = np.concatenate([strat_train_idx, rare_idx])

        test_proportion = test_ratio / val_test_ratio
        val_idx, test_idx = train_test_split(
            val_test_idx,
            test_size=test_proportion,
            random_state=random_state,
            stratify=root_labels[val_test_idx],
        )

        rare_class_names = [taxonomy.levels[0].decode(int(lbl)) for lbl in rare_labels]
        warnings.append(
            f"Rare root classes forced into training set (too few samples to stratify): "
            f"{rare_class_names}"
        )
    else:
        # Standard stratified split
        train_idx, val_test_idx = train_test_split(
            indices,
            test_size=val_test_ratio,
            random_state=random_state,
            stratify=root_labels,
        )

        test_proportion = test_ratio / val_test_ratio
        val_idx, test_idx = train_test_split(
            val_test_idx,
            test_size=test_proportion,
            random_state=random_state,
            stratify=root_labels[val_test_idx],
        )

    def make_split(idx: np.ndarray) -> DataSplit:
        return DataSplit(
            embeddings=embeddings[idx],
            labels_by_depth={d: lbls[idx] for d, lbls in labels_by_depth.items()},
            texts=[texts[i] for i in idx],
        )

    return (
        taxonomy,
        make_split(train_idx),
        make_split(val_idx),
        make_split(test_idx),
        warnings,
    )


def analyze_data(
    file_path: str | Path,
    text_column: str,
    level_columns: list[str],
) -> dict:
    """Analyze training data without embedding. Returns summary stats."""
    df = load_dataframe(file_path)
    warnings = validate_columns(df, text_column, level_columns)

    all_cols = [text_column] + level_columns
    clean_df = df.dropna(subset=all_cols).reset_index(drop=True)
    if len(clean_df) < len(df):
        warnings.append(f"{len(df) - len(clean_df)} rows have missing values")

    warnings.extend(check_hierarchy_consistency(clean_df, level_columns))

    level_stats = []
    for col in level_columns:
        counts = clean_df[col].value_counts()
        level_stats.append(
            {
                "column": col,
                "num_classes": len(counts),
                "class_counts": counts.to_dict(),
                "min_count": int(counts.min()),
                "max_count": int(counts.max()),
            }
        )

    return {
        "total_rows": len(df),
        "clean_rows": len(clean_df),
        "text_column": text_column,
        "num_levels": len(level_columns),
        "level_stats": level_stats,
        "warnings": warnings,
        "text_sample": clean_df[text_column].head(5).tolist(),
    }
