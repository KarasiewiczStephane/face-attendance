"""Tests for Docker configuration files."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


class TestDockerfile:
    """Validate Dockerfile structure."""

    def test_dockerfile_exists(self) -> None:
        """Test Dockerfile exists."""
        assert (ROOT / "Dockerfile").is_file()

    def test_dockerfile_has_multistage_build(self) -> None:
        """Test Dockerfile uses multi-stage build."""
        content = (ROOT / "Dockerfile").read_text()
        assert "AS builder" in content
        assert "COPY --from=builder" in content

    def test_dockerfile_exposes_port(self) -> None:
        """Test Dockerfile exposes port 8000."""
        content = (ROOT / "Dockerfile").read_text()
        assert "EXPOSE 8000" in content

    def test_dockerfile_has_healthcheck(self) -> None:
        """Test Dockerfile has a healthcheck."""
        content = (ROOT / "Dockerfile").read_text()
        assert "HEALTHCHECK" in content

    def test_dockerfile_copies_source(self) -> None:
        """Test Dockerfile copies src and configs."""
        content = (ROOT / "Dockerfile").read_text()
        assert "COPY src/ src/" in content
        assert "COPY configs/ configs/" in content

    def test_dockerfile_sets_unbuffered(self) -> None:
        """Test Dockerfile sets PYTHONUNBUFFERED."""
        content = (ROOT / "Dockerfile").read_text()
        assert "PYTHONUNBUFFERED=1" in content

    def test_dockerfile_has_cmd(self) -> None:
        """Test Dockerfile has CMD instruction."""
        content = (ROOT / "Dockerfile").read_text()
        assert "CMD" in content
        assert "uvicorn" in content


class TestDockerCompose:
    """Validate docker-compose.yml structure."""

    def test_compose_file_exists(self) -> None:
        """Test docker-compose.yml exists."""
        assert (ROOT / "docker-compose.yml").is_file()

    def test_compose_valid_yaml(self) -> None:
        """Test docker-compose.yml is valid YAML."""
        content = (ROOT / "docker-compose.yml").read_text()
        config = yaml.safe_load(content)
        assert isinstance(config, dict)

    def test_compose_has_service(self) -> None:
        """Test docker-compose.yml defines face-attendance service."""
        content = (ROOT / "docker-compose.yml").read_text()
        config = yaml.safe_load(content)
        assert "services" in config
        assert "face-attendance" in config["services"]

    def test_compose_port_mapping(self) -> None:
        """Test service maps port 8000."""
        content = (ROOT / "docker-compose.yml").read_text()
        config = yaml.safe_load(content)
        service = config["services"]["face-attendance"]
        assert "8000:8000" in service["ports"]

    def test_compose_volumes(self) -> None:
        """Test service has data and config volumes."""
        content = (ROOT / "docker-compose.yml").read_text()
        config = yaml.safe_load(content)
        service = config["services"]["face-attendance"]
        assert any("data" in v for v in service["volumes"])
        assert any("configs" in v for v in service["volumes"])

    def test_compose_restart_policy(self) -> None:
        """Test service has restart policy."""
        content = (ROOT / "docker-compose.yml").read_text()
        config = yaml.safe_load(content)
        service = config["services"]["face-attendance"]
        assert service["restart"] == "unless-stopped"

    def test_compose_healthcheck(self) -> None:
        """Test service has healthcheck."""
        content = (ROOT / "docker-compose.yml").read_text()
        config = yaml.safe_load(content)
        service = config["services"]["face-attendance"]
        assert "healthcheck" in service


class TestDockerignore:
    """Validate .dockerignore file."""

    def test_dockerignore_exists(self) -> None:
        """Test .dockerignore exists."""
        assert (ROOT / ".dockerignore").is_file()

    def test_dockerignore_excludes_git(self) -> None:
        """Test .dockerignore excludes .git directory."""
        content = (ROOT / ".dockerignore").read_text()
        assert ".git" in content

    def test_dockerignore_excludes_tests(self) -> None:
        """Test .dockerignore excludes tests directory."""
        content = (ROOT / ".dockerignore").read_text()
        assert "tests" in content

    def test_dockerignore_excludes_venv(self) -> None:
        """Test .dockerignore excludes virtual environments."""
        content = (ROOT / ".dockerignore").read_text()
        assert ".venv" in content or "venv" in content

    def test_dockerignore_excludes_cache(self) -> None:
        """Test .dockerignore excludes cache directories."""
        content = (ROOT / ".dockerignore").read_text()
        assert "__pycache__" in content
