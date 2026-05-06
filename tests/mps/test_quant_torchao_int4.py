"""
torchao Int4WeightOnlyConfig quantization on MPS.
MPS-native alternative to bitsandbytes 4-bit.
Requires torchao installed and ~1 GB model download on first run.
"""

import pytest
import torch
from tests.mps.helpers import MODEL_NAME, MAX_SEQ, _apply_lora, _run_training, assert_losses_finite

MPS = torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(not MPS, reason="MPS not available")

torchao = pytest.importorskip("torchao", reason="torchao not installed")


def test_torchao_int4(tmp_path):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ,
        dtype=torch.float16,
        load_in_fp8="int4_weight_only",
    )
    model = _apply_lora(model)
    losses = _run_training(model, tokenizer, fp16=True, bf16=False, output_dir=str(tmp_path))
    assert_losses_finite(losses)
