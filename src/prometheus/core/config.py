"""Configuration and directory management."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PrometheusConfig:
    """Global configuration for Prometheus."""
    base_dir: Path = field(default_factory=lambda: Path.home() / ".prometheus")
    default_embedding_model: str = "all-MiniLM-L6-v2"

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def feedback_dir(self) -> Path:
        return self.base_dir / "feedback"

    def ensure_dirs(self):
        """Create all required directories."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

    def model_path(self, model_name: str) -> Path:
        return self.models_dir / model_name

    def list_models(self) -> list[str]:
        """List all saved model names."""
        if not self.models_dir.exists():
            return []
        return sorted(
            d.name for d in self.models_dir.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()
        )
