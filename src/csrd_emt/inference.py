"""Inference uses the emotion encoder and classifier only."""

from pathlib import Path
import torch
from .models import ResearchEmT, TEMPERATURE


def export_student(student, path):
    device = next(student.parameters()).device
    inference = ResearchEmT("erm").to(device).eval()
    state = {name: student.state_dict()[name] for name in inference.state_dict()}
    inference.load_state_dict(state, strict=True)
    torch.save(
        {
            "model": {k: v.cpu() for k, v in state.items()},
            "classes": ["negative", "neutral", "positive"],
            "input_shape": [36, 62, 5],
            "cosine_temperature": TEMPERATURE,
        },
        path,
    )
    return inference


def load_student(checkpoint, device="cpu"):
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=True)
    if (
        payload["input_shape"] != [36, 62, 5]
        or payload["cosine_temperature"] != TEMPERATURE
    ):
        raise ValueError("Incompatible input shape or cosine temperature")
    model = ResearchEmT("erm").to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    return model
