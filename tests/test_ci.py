"""Tests for CI/CD configuration files."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


class TestCIWorkflow:
    """Validate GitHub Actions CI workflow."""

    def test_ci_file_exists(self) -> None:
        """Test CI workflow file exists."""
        assert (ROOT / ".github" / "workflows" / "ci.yml").is_file()

    def test_ci_valid_yaml(self) -> None:
        """Test CI workflow is valid YAML."""
        content = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        config = yaml.safe_load(content)
        assert isinstance(config, dict)

    def test_ci_triggers(self) -> None:
        """Test CI triggers on push and PR to main/master."""
        content = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        config = yaml.safe_load(content)
        # YAML parses "on" as True (boolean), so access via True key
        triggers = config.get("on") or config.get(True)
        assert "push" in triggers
        assert "pull_request" in triggers
        push_branches = triggers["push"]["branches"]
        assert "main" in push_branches or "master" in push_branches

    def test_ci_has_lint_job(self) -> None:
        """Test CI has a lint job."""
        content = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        config = yaml.safe_load(content)
        assert "lint" in config["jobs"]

    def test_ci_has_test_job(self) -> None:
        """Test CI has a test job."""
        content = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        config = yaml.safe_load(content)
        assert "test" in config["jobs"]

    def test_ci_has_docker_job(self) -> None:
        """Test CI has a Docker build job."""
        content = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        config = yaml.safe_load(content)
        assert "build-docker" in config["jobs"]

    def test_ci_test_depends_on_lint(self) -> None:
        """Test that test job depends on lint job."""
        content = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        config = yaml.safe_load(content)
        assert "lint" in config["jobs"]["test"]["needs"]

    def test_ci_docker_depends_on_test(self) -> None:
        """Test that docker job depends on test job."""
        content = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        config = yaml.safe_load(content)
        assert "test" in config["jobs"]["build-docker"]["needs"]

    def test_ci_uses_python_311(self) -> None:
        """Test CI uses Python 3.11."""
        content = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "3.11" in content

    def test_ci_runs_coverage(self) -> None:
        """Test CI runs tests with coverage."""
        content = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "--cov=src" in content


class TestPyprojectConfig:
    """Validate pyproject.toml configuration."""

    def test_pyproject_exists(self) -> None:
        """Test pyproject.toml exists."""
        assert (ROOT / "pyproject.toml").is_file()

    def test_pyproject_has_coverage_threshold(self) -> None:
        """Test pyproject.toml has coverage fail_under."""
        content = (ROOT / "pyproject.toml").read_text()
        assert "fail_under = 80" in content

    def test_pyproject_has_ruff_config(self) -> None:
        """Test pyproject.toml has ruff configuration."""
        content = (ROOT / "pyproject.toml").read_text()
        assert "[tool.ruff]" in content

    def test_pyproject_has_pytest_config(self) -> None:
        """Test pyproject.toml has pytest configuration."""
        content = (ROOT / "pyproject.toml").read_text()
        assert "[tool.pytest.ini_options]" in content
