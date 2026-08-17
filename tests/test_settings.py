"""
tests/test_settings.py
──────────────────────
Smoke tests for project bootstrap (updated for Chroma, no Qdrant).
Verifies that:
  1. The settings singleton loads without error.
  2. Every expected field exists with a sensible default.
  3. The use_groq property works correctly.
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
    from importlib import reload
    import config.settings as settings_module

    original = os.environ.get("GROQ_API_KEY", "")
    os.environ.pop("GROQ_API_KEY", None)

    reload(settings_module)
    s = settings_module.Settings()

    # Updated defaults (no Qdrant)
    assert s.groq_model == "openai/gpt-oss-120b"
    assert s.ollama_model == "mistral"
    assert s.embed_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.reranker_model == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert s.dense_top_k == 20
    assert s.bm25_top_k == 20
    assert s.rrf_k == 60
    assert s.rerank_top_n == 5
    assert s.api_port == 8000

    # Restore
    if original:
        os.environ["GROQ_API_KEY"] = original


# ── 2. use_groq property ─────────────────────────────────────────────────────────

def test_use_groq_false_when_no_key():
    from config.settings import Settings
    s = Settings(GROQ_API_KEY="")
    assert s.use_groq is False


def test_use_groq_true_when_key_present():
    from config.settings import Settings
    s = Settings(GROQ_API_KEY="gsk_test_key_123")
    assert s.use_groq is True


# ── 3. Directory layout ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("folder", [
    "config", "ingestion", "retrieval", "src", "tests",
    "data/pdfs",
])
def test_directory_exists(folder):
    assert (PROJECT_ROOT / folder).is_dir(), f"Missing directory: {folder}"


@pytest.mark.parametrize("pkg", [
    "config/__init__.py",
    "src/__init__.py",
    "src/retrieval/__init__.py",
    "src/generation/__init__.py",
    "src/ingestion/__init__.py",
    "tests/__init__.py",
])
def test_init_files_exist(pkg):
    assert (PROJECT_ROOT / pkg).is_file(), f"Missing __init__.py: {pkg}"


# ── 4. Required top-level files ───────────────────────────────────────────────────

@pytest.mark.parametrize("fname", [
    "requirements.txt",
    ".env.example",
    "README.md",
    "config/settings.py",
])
def test_required_files_exist(fname):
    assert (PROJECT_ROOT / fname).is_file(), f"Missing file: {fname}"


# ── 5. requirements.txt sanity ────────────────────────────────────────────────────

def test_requirements_contains_key_packages():
    req = (PROJECT_ROOT / "requirements.txt").read_text()
    # Updated: chromadb instead of qdrant-client
    for pkg in ["pymupdf", "chromadb", "sentence-transformers",
                "rank-bm25", "fastapi", "streamlit",
                "pytest"]:
        assert pkg in req, f"requirements.txt missing: {pkg}"
