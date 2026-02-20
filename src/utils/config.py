"""Configuration management for the face attendance system.

Loads YAML configuration and provides type-safe access to settings
via Pydantic BaseSettings integration.
"""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

from .logger import setup_logger

logger = setup_logger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment and config file."""

    config_path: Path = Field(default=Path("configs/config.yaml"))

    model_config = {"env_prefix": "FACE_ATTENDANCE_"}

    def load_config(self) -> dict:
        """Load and return the YAML configuration.

        Returns:
            Parsed YAML configuration as a dictionary.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        if not self.config_path.exists():
            logger.warning("Config file not found at %s, using defaults", self.config_path)
            return _default_config()
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded from %s", self.config_path)
        return config


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton.

    Returns:
        Settings instance.
    """
    return Settings()


def _default_config() -> dict:
    """Return default configuration when config file is missing.

    Returns:
        Dictionary with sensible default values.
    """
    return {
        "face_detection": {
            "min_face_size": 20,
            "thresholds": [0.6, 0.7, 0.7],
            "device": "cpu",
        },
        "embedding": {
            "model": "vggface2",
            "image_size": 160,
            "embedding_dim": 512,
        },
        "matching": {
            "similarity_threshold": 0.7,
            "algorithm": "cosine",
        },
        "liveness": {
            "blink_threshold": 0.25,
            "texture_threshold": 0.5,
            "challenge_timeout": 10,
        },
        "attendance": {
            "dedup_window_hours": 4,
            "work_start": "09:00",
            "work_end": "18:00",
            "late_threshold_minutes": 15,
        },
        "privacy": {
            "retention_days": 365,
            "audit_enabled": True,
        },
        "database": {
            "path": "data/attendance.db",
        },
    }
