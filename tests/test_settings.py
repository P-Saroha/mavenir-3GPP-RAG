"""
tests/test_settings.py
──────────────────────
Smoke tests for project bootstrap.
Verifies that:
  1. The settings singleton loads without error.
  2. Every expected field exists with a sensible default.
  3. The use_grok property works correctly.
  4. Key directories exist.
  5. Required top-level files exist.
"""

import os
from pathlib import Path

import pytest

# Project root is two levels up from this file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── 1. Settings load ─────────────────────────────────────────────────────────────

def test_settings_import():
    """Settings singleton imports without raising."""
    from config.settings import settings
    assert settings is not None


def test_settings_defaults():
    """All fields have sensible defaults even with no .env present."""
    # Temporarily unset GROK_API_KEY so we test pure defaults.
    from importlib import reload
    import config.settings as settings_module

    original = os.environ.get("GROK_API_KEY", "")
    os.environ.pop("GROK_API_KEY", None)

    reload(settings_module)
    s = settings_module.Settings()

    assert s.grok_model == "grok-3-mini"
    assert s.ollama_model == "mistral"
    assert s.embed_model == "nomic-ai/nomic-embed-text-v1.5"
    assert s.reranker_model == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert s.qdrant_collection == "3gpp_r17"
    assert s.dense_top_k == 20
    assert s.bm25_top_k == 20
    assert s.rrf_k == 60
    assert s.rerank_top_n == 5
    assert s.api_port == 8000

    # Restore
    if original:
        os.environ["GROK_API_KEY"] = original


# ── 2. use_grok property ─────────────────────────────────────────────────────────

def test_use_grok_false_when_no_key():
    from config.settings import Settings
    s = Settings(GROK_API_KEY="")
    assert s.use_grok is False


def test_use_grok_true_when_key_present():
    from config.settings import Settings
    s = Settings(GROK_API_KEY="xai-test-key-123")
    assert s.use_grok is True


# ── 3. Directory layout ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("folder", [
    "config", "ingestion", "retrieval", "api", "ui", "tests",
    "data/pdfs", "data/qdrant_storage",
])
def test_directory_exists(folder):
    assert (PROJECT_ROOT / folder).is_dir(), f"Missing directory: {folder}"


@pytest.mark.parametrize("pkg", [
    "config/__init__.py",
    "ingestion/__init__.py",
    "retrieval/__init__.py",
    "api/__init__.py",
    "ui/__init__.py",
    "tests/__init__.py",
])
def test_init_files_exist(pkg):
    assert (PROJECT_ROOT / pkg).is_file(), f"Missing __init__.py: {pkg}"


# ── 4. Required top-level files ───────────────────────────────────────────────────

@pytest.mark.parametrize("fname", [
    "requirements.txt",
    "pyproject.toml",
    ".env.example",
    "setup.ps1",
    "setup.sh",
    "README.md",
    "config/settings.py",
])
def test_required_files_exist(fname):
    assert (PROJECT_ROOT / fname).is_file(), f"Missing file: {fname}"


# ── 5. requirements.txt sanity ────────────────────────────────────────────────────

def test_requirements_contains_key_packages():
    req = (PROJECT_ROOT / "requirements.txt").read_text()
    for pkg in ["PyMuPDF", "qdrant-client", "sentence-transformers",
                "rank-bm25", "openai", "fastapi", "streamlit",
                "pydantic-settings", "pytest"]:
        assert pkg in req, f"requirements.txt missing: {pkg}"
