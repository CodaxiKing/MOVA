"""Cheap checks against the real Wan2.1-VACE-1.3B files, only where the pinned revision is fully cached.

Loads configs and the tokenizer (a few MB), never the 1.3B transformer or the UMT5 encoder.
"""

import pytest

from common.config import load_config
from models.registry import create_model


def _cached_settings():
    model = create_model("wan", load_config("configs/baseline.yaml"))
    try:
        status = model.weights_status()
    except Exception:  # offline without a saved manifest
        return None
    if not status.complete:
        return None
    from inference.baseline_vace import BaselineSettings

    return BaselineSettings(revision=status.revision)


SETTINGS = _cached_settings()
pytestmark = pytest.mark.skipif(SETTINGS is None, reason="Wan2.1-VACE-1.3B not fully cached")


def test_umt5_tokenizer_loads_offline():
    """transformers 5 AutoTokenizer needs a model config; the repo's tokenizer/ has none (EXP-001 failure)."""
    from inference.baseline_vace import load_tokenizer

    tok = load_tokenizer(SETTINGS)
    ids = tok(["A person dancing"], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    assert ids.input_ids.shape == (1, 512) and int(ids.attention_mask.sum()) > 1
