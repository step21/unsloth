"""
4-bit FP4 quantization (bitsandbytes) on MPS.
bitsandbytes does not support MPS as of 2026 — this test will be skipped
until bnb adds MPS support.
Requires ~1 GB model download on first run.
"""

import pytest
import torch
from tests.mps.helpers import MODEL_NAME, MAX_SEQ, _apply_lora, _run_training, assert_losses_finite

MPS = torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(not MPS, reason="MPS not available")


def _bnb_supports_mps():
    try:
        import bitsandbytes as bnb
        return getattr(bnb, "HAS_MPS", False)
    except ImportError:
        return False


@pytest.mark.skipif(not _bnb_supports_mps(), reason="bitsandbytes does not support MPS")
def test_4bit_fp4(tmp_path):
    from unsloth import FastLanguageModel
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ,
        dtype=None,
        load_in_4bit=True,
        quantization_config=bnb_config,
    )
    model = _apply_lora(model)
    losses = _run_training(model, tokenizer, fp16=True, bf16=False, output_dir=str(tmp_path))
    assert_losses_finite(losses)
