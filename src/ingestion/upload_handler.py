"""
src/ingestion/upload_handler.py
--------------------------------
Handles PDF upload, duplicate detection, and ingestion status tracking.

Features:
- SHA256 file hash for duplicate detection
- Upload status tracking (uploaded, already_indexed, ingestion_started, completed, failed)
- Metadata preservation (source_type = "uploaded" for user PDFs vs "3gpp_official" for corpus)

Usage:
    handler = UploadHandler()
    status = handler.check_duplicate("myfile.pdf")
    handler.mark_uploaded("myfile.pdf", sha256_hash)
    handler.mark_completed("myfile.pdf")
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

# ── paths ──────────────────────────────────────────────────────────────────────
UPLOAD_REGISTRY_PATH = Path("data/.upload_registry.json")
PDF_DIR = Path("data/pdfs")


UploadStatus = Literal["uploaded", "already_indexed", "ingestion_started", "ingestion_completed", "ingestion_failed"]


@dataclass
class UploadRecord:
    filename: str
    file_hash: str  # SHA256
    status: UploadStatus
    timestamp: str
    error_message: str | None = None


class UploadHandler:
    """Manages PDF upload tracking and duplicate detection."""

    def __init__(self):
        self.registry_path = UPLOAD_REGISTRY_PATH
        self.pdf_dir = PDF_DIR
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()

    def _load_registry(self) -> dict[str, UploadRecord]:
        """Load existing upload registry."""
        if not self.registry_path.exists():
            return {}
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            return {
                k: UploadRecord(**v) for k, v in data.items()
            }
        except Exception:
            return {}

    def _save_registry(self):
        """Save registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: asdict(v) for k, v in self.registry.items()}
        self.registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def check_duplicate(self, filename: str) -> tuple[bool, str | None]:
        """
        Check if a file with the same name and hash has already been processed.

        Returns:
            (is_duplicate, previous_status)
        """
        if filename in self.registry:
            return True, self.registry[filename].status
        return False, None

    def mark_uploaded(self, filename: str, file_hash: str):
        """Mark a file as uploaded."""
        self.registry[filename] = UploadRecord(
            filename=filename,
            file_hash=file_hash,
            status="uploaded",
            timestamp=datetime.now().isoformat(),
        )
        self._save_registry()

    def mark_ingestion_started(self, filename: str):
        """Mark ingestion as started."""
        if filename in self.registry:
            self.registry[filename].status = "ingestion_started"
            self._save_registry()

    def mark_ingestion_completed(self, filename: str):
        """Mark ingestion as completed."""
        if filename in self.registry:
            self.registry[filename].status = "ingestion_completed"
            self.registry[filename].error_message = None
            self._save_registry()

    def mark_ingestion_failed(self, filename: str, error: str):
        """Mark ingestion as failed with an error message."""
        if filename in self.registry:
            self.registry[filename].status = "ingestion_failed"
            self.registry[filename].error_message = error
            self._save_registry()

    def get_status(self, filename: str) -> UploadRecord | None:
        """Get the status of an uploaded file."""
        return self.registry.get(filename)
