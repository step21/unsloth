"""
No quantization, bf16 weights — bf16 LoRA baseline on MPS.
MPS supports bf16 from PyTorch 2.4.
Requires ~1 GB model download on first run (unsloth/Llama-3.2-1B-Instruct).
"""

import pytest
import torch
from unsloth import FastLanguageModel
from tests.mps.helpers import MODEL_NAME, MAX_SEQ, _apply_lora, _run_training, assert_losses_finite

MPS = torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(not MPS, reason="MPS not available")


def test_no_quant_bf16(tmp_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    model = _apply_lora(model)
    losses = _run_training(model, tokenizer, fp16=False, bf16=True, output_dir=str(tmp_path))
    assert_losses_finite(losses)
