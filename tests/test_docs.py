"""Tests for documentation completeness."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestReadme:
    """Validate README.md content."""

    def test_readme_exists(self) -> None:
        """Test README.md exists."""
        assert (ROOT / "README.md").is_file()

    def test_readme_has_title(self) -> None:
        """Test README has project title."""
        content = (ROOT / "README.md").read_text()
        assert "Face Recognition Attendance System" in content

    def test_readme_has_features(self) -> None:
        """Test README lists features."""
        content = (ROOT / "README.md").read_text()
        assert "## Features" in content
        assert "Face Detection" in content
        assert "Face Recognition" in content
        assert "Anti-Spoofing" in content

    def test_readme_has_quick_start(self) -> None:
        """Test README has quick start section."""
        content = (ROOT / "README.md").read_text()
        assert "## Quick Start" in content
        assert "pip install" in content

    def test_readme_has_api_endpoints(self) -> None:
        """Test README documents API endpoints."""
        content = (ROOT / "README.md").read_text()
        assert "/enroll" in content
        assert "/verify" in content
        assert "/attendance" in content
        assert "/health" in content

    def test_readme_has_docker(self) -> None:
        """Test README has Docker section."""
        content = (ROOT / "README.md").read_text()
        assert "## Docker" in content
        assert "docker compose" in content

    def test_readme_has_privacy(self) -> None:
        """Test README has privacy section."""
        content = (ROOT / "README.md").read_text()
        assert "Privacy" in content
        assert "GDPR" in content

    def test_readme_has_architecture(self) -> None:
        """Test README has architecture section."""
        content = (ROOT / "README.md").read_text()
        assert "## Architecture" in content

    def test_readme_has_project_structure(self) -> None:
        """Test README has project structure."""
        content = (ROOT / "README.md").read_text()
        assert "## Project Structure" in content


class TestAPIDocs:
    """Validate API documentation."""

    def test_api_docs_exist(self) -> None:
        """Test API docs file exists."""
        assert (ROOT / "docs" / "api.md").is_file()

    def test_api_docs_cover_all_endpoints(self) -> None:
        """Test API docs cover all endpoints."""
        content = (ROOT / "docs" / "api.md").read_text()
        endpoints = [
            "/health",
            "/enroll",
            "/verify",
            "/attendance/today",
            "/attendance/report",
            "/person/",
            "/persons",
            "/audit",
        ]
        for endpoint in endpoints:
            assert endpoint in content, f"Missing endpoint: {endpoint}"

    def test_api_docs_have_examples(self) -> None:
        """Test API docs include curl examples."""
        content = (ROOT / "docs" / "api.md").read_text()
        assert "curl" in content


class TestArchitectureDocs:
    """Validate architecture documentation."""

    def test_architecture_docs_exist(self) -> None:
        """Test architecture docs file exists."""
        assert (ROOT / "docs" / "architecture.md").is_file()

    def test_architecture_docs_cover_components(self) -> None:
        """Test architecture docs describe all components."""
        content = (ROOT / "docs" / "architecture.md").read_text()
        components = [
            "Face Detection",
            "Embedding",
            "Liveness",
            "Matching",
            "Attendance",
            "Report",
            "Privacy",
            "REST API",
        ]
        for component in components:
            assert component in content, f"Missing component: {component}"

    def test_architecture_docs_have_data_flow(self) -> None:
        """Test architecture docs describe data flow."""
        content = (ROOT / "docs" / "architecture.md").read_text()
        assert "Data Flow" in content
        assert "Enrollment" in content
        assert "Verification" in content


class TestSampleData:
    """Validate sample data directory."""

    def test_sample_dir_exists(self) -> None:
        """Test sample data directory exists."""
        assert (ROOT / "data" / "sample").is_dir()

    def test_sample_readme_exists(self) -> None:
        """Test sample data has README."""
        assert (ROOT / "data" / "sample" / "README.md").is_file()
