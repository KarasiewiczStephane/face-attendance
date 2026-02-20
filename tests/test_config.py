"""Tests for configuration management."""

from pathlib import Path

import pytest
import yaml

from src.utils.config import Settings, _default_config


@pytest.fixture
def valid_config_file(tmp_path: Path) -> Path:
    """Create a valid config YAML file for testing."""
    config = {
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
        "database": {
            "path": "data/test.db",
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


class TestSettings:
    """Tests for Settings class."""

    def test_load_config_valid(self, valid_config_file: Path) -> None:
        """Test loading a valid YAML config file."""
        settings = Settings(config_path=valid_config_file)
        config = settings.load_config()

        assert config["face_detection"]["min_face_size"] == 20
        assert config["embedding"]["model"] == "vggface2"
        assert config["matching"]["similarity_threshold"] == 0.7

    def test_load_config_missing_file(self, tmp_path: Path) -> None:
        """Test loading config when file is missing returns defaults."""
        missing_path = tmp_path / "nonexistent.yaml"
        settings = Settings(config_path=missing_path)
        config = settings.load_config()

        assert "face_detection" in config
        assert "database" in config
        assert config["database"]["path"] == "data/attendance.db"

    def test_load_config_preserves_all_sections(self, valid_config_file: Path) -> None:
        """Test that all config sections are accessible."""
        settings = Settings(config_path=valid_config_file)
        config = settings.load_config()

        assert "face_detection" in config
        assert "embedding" in config
        assert "matching" in config
        assert "database" in config

    def test_load_config_thresholds_list(self, valid_config_file: Path) -> None:
        """Test that thresholds are loaded as a list."""
        settings = Settings(config_path=valid_config_file)
        config = settings.load_config()

        thresholds = config["face_detection"]["thresholds"]
        assert isinstance(thresholds, list)
        assert len(thresholds) == 3

    def test_default_config_path(self) -> None:
        """Test the default config path is set correctly."""
        settings = Settings()
        assert settings.config_path == Path("configs/config.yaml")


class TestDefaultConfig:
    """Tests for default configuration fallback."""

    def test_default_config_has_all_sections(self) -> None:
        """Test default config contains all required sections."""
        config = _default_config()
        required_sections = [
            "face_detection",
            "embedding",
            "matching",
            "liveness",
            "attendance",
            "privacy",
            "database",
        ]
        for section in required_sections:
            assert section in config, f"Missing section: {section}"

    def test_default_config_values(self) -> None:
        """Test specific default values."""
        config = _default_config()
        assert config["embedding"]["image_size"] == 160
        assert config["embedding"]["embedding_dim"] == 512
        assert config["attendance"]["dedup_window_hours"] == 4
        assert config["privacy"]["retention_days"] == 365
