"""Cross-subject reference distillation for EEG emotion recognition."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from .inference import load_student

__all__ = ["load_student"]
