"""
src/utils/model_cache.py
------------------------
Download and cache embedding + reranker models locally on first use.
Subsequent runs load from disk (no re-download, no re-initialization).

Models are cached in HF_HOME environment variable, or ~/.cache/huggingface by default.

Usage:
    python -m src.utils.model_cache --download  # Pre-download all models
    python src/utils/model_cache.py --download  # same
"""

import os
import sys
from pathlib import Path

# Set Hugging Face cache directory (optional — defaults to ~/.cache/huggingface)
# This ensures models are cached in a predictable location
HF_HOME = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ["HF_HOME"] = HF_HOME

from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def download_embedding_model(verbose: bool = True):
    """Download embedding model once and cache it."""
    if verbose:
        print(f"[Model Cache] Downloading embedding model: {EMBEDDING_MODEL}")
        print(f"              Cache location: {HF_HOME}")
    
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    
    if verbose:
        cache_size = sum(
            f.stat().st_size for f in Path(HF_HOME).rglob("*") if f.is_file()
        ) // (1024 ** 2)
        print(f"[Model Cache] Downloaded successfully. Total cache: {cache_size} MB")
    
    return model


def download_reranker_model(verbose: bool = True):
    """Download reranker model once and cache it."""
    if verbose:
        print(f"[Model Cache] Downloading reranker model: {RERANKER_MODEL}")
        print(f"              Cache location: {HF_HOME}")
    
    model = SentenceTransformer(RERANKER_MODEL, trust_remote_code=True)
    
    if verbose:
        cache_size = sum(
            f.stat().st_size for f in Path(HF_HOME).rglob("*") if f.is_file()
        ) // (1024 ** 2)
        print(f"[Model Cache] Downloaded successfully. Total cache: {cache_size} MB")
    
    return model


def ensure_models_cached(verbose: bool = True):
    """Ensure both models are downloaded and cached."""
    download_embedding_model(verbose=verbose)
    download_reranker_model(verbose=verbose)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Pre-download and cache embedding + reranker models"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download all models (or just call this script to do the same)"
    )
    args = parser.parse_args()
    
    print(f"HuggingFace cache: {HF_HOME}\n")
    ensure_models_cached(verbose=True)
    print("\n[OK] All models cached and ready for production use.")
