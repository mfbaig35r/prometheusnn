"""Configuration and directory management."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from prometheus.core.mcp_safety import ToolValidationError

# A model name is one path segment: letters, digits, dot, underscore, hyphen; no leading dot.
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_model_name(model_name: str) -> str:
    """Reject anything that is not a single, plain directory name (no '/', '..', absolute paths)."""
    if not isinstance(model_name, str) or not _MODEL_NAME.match(model_name) or ".." in model_name:
        raise ToolValidationError(
            f"Invalid model name {model_name!r}: use letters, digits, '.', '_' or '-' "
            "(max 128 characters, no path separators)",
            field="model_name",
        )
    return model_name


def _contained(base: Path, child: Path) -> Path:
    """Resolve child and require it to stay inside base (defence in depth after validation)."""
    base_r, child_r = base.resolve(), child.resolve()
    if child_r != base_r and base_r not in child_r.parents:
        raise ToolValidationError(f"Path escapes {base}", field="model_name")
    return child_r


@dataclass
class PrometheusConfig:
    """Global configuration for Prometheus."""

    base_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("PROMETHEUS_HOME", Path.home() / ".prometheus"))
    )
    default_embedding_model: str = field(
        default_factory=lambda: os.environ.get("PROMETHEUS_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def feedback_dir(self) -> Path:
        return self.base_dir / "feedback"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def exports_dir(self) -> Path:
        override = os.environ.get("PROMETHEUS_EXPORT_DIR")
        return Path(override) if override else self.base_dir / "exports"

    def ensure_dirs(self):
        """Create all required directories."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def model_path(self, model_name: str) -> Path:
        """Directory for a model; always inside models_dir."""
        return _contained(self.models_dir, self.models_dir / validate_model_name(model_name))

    def feedback_path(self, model_name: str) -> Path:
        """Feedback directory for a model; always inside feedback_dir."""
        return _contained(self.feedback_dir, self.feedback_dir / validate_model_name(model_name))

    def export_path(self, output_path: str) -> Path:
        """Destination for export_model; always inside exports_dir.

        A relative path is taken under exports_dir; an absolute path must already point inside
        it. Set PROMETHEUS_EXPORT_DIR to export somewhere else.
        """
        if not isinstance(output_path, str) or not output_path.strip():
            raise ToolValidationError("output_path is required", field="output_path")
        candidate = Path(output_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.exports_dir / candidate
        try:
            return _contained(self.exports_dir, candidate)
        except ToolValidationError as exc:
            raise ToolValidationError(
                f"output_path must be inside the export directory {self.exports_dir} "
                "(set PROMETHEUS_EXPORT_DIR to change it)",
                field="output_path",
            ) from exc

    def list_models(self) -> list[str]:
        """List all saved model names."""
        if not self.models_dir.exists():
            return []
        return sorted(
            d.name
            for d in self.models_dir.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()
        )
