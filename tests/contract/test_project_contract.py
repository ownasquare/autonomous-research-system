from pathlib import Path


def test_required_public_files_exist(project_root: Path) -> None:
    required = {
        "README.md",
        "LICENSE",
        "SECURITY.md",
        "CONTRIBUTING.md",
        ".env.example",
        "Makefile",
        "pyproject.toml",
    }
    assert required <= {path.name for path in project_root.iterdir()}


def test_source_layout_is_installable(project_root: Path) -> None:
    package = project_root / "src" / "research_system"
    assert (package / "__init__.py").is_file()
    assert (package / "py.typed").is_file()


def test_docker_context_excludes_local_secret_files(project_root: Path) -> None:
    patterns = {
        line.strip()
        for line in (project_root / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {".env", ".env.*"} <= patterns
