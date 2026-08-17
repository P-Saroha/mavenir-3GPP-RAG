#!/usr/bin/env python
"""
setup_models.py
---------------
One-time setup: Download and cache all models locally.

This MUST be run once before the first PDF ingestion.
After this, PDF ingestion will be 10x-100x faster (no re-download, instant model load).

Run:
    python setup_models.py

Time breakdown:
  - First run (this script):   2-3 minutes (downloads ~500 MB)
  - Subsequent ingestions:     5-10 seconds (loads from cache)
  - Without caching:           30-60 minutes per ingestion (CPU-based encoding)
"""

import sys
import time
from pathlib import Path

print("=" * 70)
print("3GPP RAG Chatbot — Model Cache Setup")
print("=" * 70)
print()

# Add workspace to path so we can import
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.model_cache import ensure_models_cached

print("This will download ~500 MB of models to your local cache.")
print("After this, all future ingestions will be 10x faster.")
print()

try:
    t0 = time.time()
    ensure_models_cached(verbose=True)
    elapsed = time.time() - t0
    
    print()
    print("=" * 70)
    print(f"[OK] Setup complete in {elapsed:.1f} seconds")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Upload a PDF in the Streamlit app")
    print("  2. Ingestion will now take 5-10 seconds (not 30-60 minutes)")
    print("  3. Models are cached and will be reused forever")
    print()
    
except Exception as e:
    print()
    print("[ERROR] Failed to download models:")
    print(f"  {e}")
    print()
    print("Troubleshooting:")
    print("  - Check internet connection")
    print("  - Try: pip install --upgrade sentence-transformers")
    print("  - Try: pip install --upgrade huggingface-hub")
    sys.exit(1)
