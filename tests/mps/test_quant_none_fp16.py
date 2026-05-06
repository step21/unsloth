"""
No quantization, fp16 weights — full-precision LoRA baseline on MPS.
Requires ~1 GB model download on first run (unsloth/Llama-3.2-1B-Instruct).
"""

import pytest
import torch
from unsloth import FastLanguageModel
from tests.mps.helpers import MODEL_NAME, MAX_SEQ, _apply_lora, _run_training, assert_losses_finite

MPS = torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(not MPS, reason="MPS not available")


def test_no_quant_fp16(tmp_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ,
        dtype=torch.float16,
        load_in_4bit=False,
    )
    model = _apply_lora(model)
    losses = _run_training(model, tokenizer, fp16=True, bf16=False, output_dir=str(tmp_path))
    assert_losses_finite(losses)
