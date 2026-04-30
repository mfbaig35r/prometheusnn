"""Column profiling: dtype, cardinality, kind classification."""

from __future__ import annotations

import re

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from prometheus.intake.models import ColumnKind, ColumnProfile

_TEXT_LENGTH_THRESHOLD = 15  # avg chars > this → likely free text, not a label
_CATEGORICAL_CARDINALITY_RATIO = 0.5  # cardinality / rows < this → categorical
_IDENTIFIER_PATTERN = re.compile(r"^(id|uuid|guid|key|hash|sku|code)$", re.IGNORECASE)


def profile_columns(df: pd.DataFrame, sample_size: int = 5) -> list[ColumnProfile]:
    """Profile every column in the DataFrame."""
    profiles = []
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        cardinality = int(non_null.nunique())
        null_rate = float(series.isna().mean())
        kind = classify_column_kind(series, name=col)

        avg_len: float | None = None
        median_len: float | None = None
        if kind in (ColumnKind.TEXT, ColumnKind.CATEGORICAL) and len(non_null) > 0:
            lengths = non_null.astype(str).str.len()
            avg_len = float(lengths.mean())
            median_len = float(lengths.median())

        sample_values = _take_samples(non_null, sample_size)

        profiles.append(
            ColumnProfile(
                name=str(col),
                kind=kind,
                dtype=str(series.dtype),
                cardinality=cardinality,
                null_rate=null_rate,
                sample_values=sample_values,
                avg_text_length=avg_len,
                median_text_length=median_len,
            )
        )
    return profiles


def classify_column_kind(series: pd.Series, name: str | None = None) -> ColumnKind:
    """Heuristically classify a column's role."""
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return ColumnKind.UNKNOWN

    if is_datetime64_any_dtype(series):
        return ColumnKind.DATETIME

    if name and _IDENTIFIER_PATTERN.match(name):
        if non_null.nunique() == n:
            return ColumnKind.IDENTIFIER

    if is_numeric_dtype(series):
        return ColumnKind.NUMERIC

    str_series = non_null.astype(str)
    if _looks_like_datetime(str_series):
        return ColumnKind.DATETIME

    cardinality = int(str_series.nunique())
    avg_len = float(str_series.str.len().mean())
    cardinality_ratio = cardinality / n

    if cardinality == n and n > 10 and avg_len <= _TEXT_LENGTH_THRESHOLD:
        return ColumnKind.IDENTIFIER

    if avg_len > _TEXT_LENGTH_THRESHOLD and cardinality_ratio > _CATEGORICAL_CARDINALITY_RATIO:
        return ColumnKind.TEXT

    if cardinality_ratio < _CATEGORICAL_CARDINALITY_RATIO or cardinality < 100:
        return ColumnKind.CATEGORICAL

    return ColumnKind.TEXT


def _take_samples(series: pd.Series, n: int) -> list[str]:
    if len(series) == 0:
        return []
    head = series.head(n).astype(str).tolist()
    return [s if len(s) <= 80 else s[:77] + "..." for s in head]


def _looks_like_datetime(str_series: pd.Series) -> bool:
    sample = str_series.head(20)
    if not sample.str.contains(r"[-/:]", regex=True).any():
        return False
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() > 0.8
    except (ValueError, TypeError):
        return False
