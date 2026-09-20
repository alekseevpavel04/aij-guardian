"""Environment setup, imported first by every entrypoint.

Caches and model directories live on D (C has ~30 GB free), so the variables that decide where
libraries download and write must be set before torch, transformers or huggingface_hub are imported.
Console output is forced to UTF-8: the transcripts contain ⟦…⟧ markers and Cyrillic, and the default
Windows console codepage cannot encode them.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Set variables from .env without overriding what is already in the environment."""
    path = path or ROOT / ".env"
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
        os.environ.setdefault(key.strip(), value.strip())
    return values


def setup(offline: bool = False) -> None:
    load_dotenv()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if offline:
        # Submission runtime has no network: fail loudly on a missing local file instead of hanging.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
