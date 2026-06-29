"""
download_models.py — One-time utility to download and save HuggingFace models
to local disk so the pipeline can run without network access.

Usage:
    python download_models.py

Saves:
    models/all-MiniLM-L6-v2/          (SentenceTransformer embedding model)
    models/cross-encoder-ms-marco/     (CrossEncoder reranking model)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"


def main():
    print("=" * 60)
    print("Downloading models to local disk...")
    print("=" * 60)

    # ── 1. SentenceTransformer ────────────────────────────────────────────
    st_dir = MODELS_DIR / "all-MiniLM-L6-v2"
    if st_dir.exists() and any(st_dir.iterdir()):
        print(f"[SKIP] SentenceTransformer already exists at {st_dir}")
    else:
        print(f"[DOWNLOAD] SentenceTransformer -> {st_dir}")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            model.save(str(st_dir))
            print(f"[OK] Saved SentenceTransformer to {st_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to download SentenceTransformer: {e}")
            sys.exit(1)

    # ── 2. CrossEncoder ───────────────────────────────────────────────────
    ce_dir = MODELS_DIR / "cross-encoder-ms-marco"
    if ce_dir.exists() and any(ce_dir.iterdir()):
        print(f"[SKIP] CrossEncoder already exists at {ce_dir}")
    else:
        print(f"[DOWNLOAD] CrossEncoder -> {ce_dir}")
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
            model.save(str(ce_dir))
            print(f"[OK] Saved CrossEncoder to {ce_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to download CrossEncoder: {e}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("All models saved locally. No network access needed at runtime.")
    print("=" * 60)


if __name__ == "__main__":
    main()
