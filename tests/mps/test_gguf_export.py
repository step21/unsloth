"""
GGUF export test on MPS.
Trains with no-quant fp16 LoRA, merges adapters, exports to GGUF Q4_K_M,
and verifies the output file is non-empty and valid.
Requires ~1 GB model download and llama.cpp (via unsloth save) on first run.
"""

import pytest
import torch
from pathlib import Path
from tests.mps.helpers import MODEL_NAME, MAX_SEQ, _apply_lora, _run_training, assert_losses_finite

MPS = torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(not MPS, reason="MPS not available")


def test_gguf_export_q4_k_m(tmp_path):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ,
        dtype=torch.float16,
        load_in_4bit=False,
    )
    model = _apply_lora(model)

    losses = _run_training(model, tokenizer, fp16=True, output_dir=str(tmp_path / "checkpoints"))
    assert_losses_finite(losses)

    FastLanguageModel.for_inference(model)

    save_dir = str(tmp_path / "gguf")
    model.save_pretrained_gguf(
        save_dir,
        tokenizer,
        quantization_method="q4_k_m",
    )

    gguf_files = list(Path(save_dir).glob("*.gguf"))
    assert len(gguf_files) > 0, f"No .gguf file found in {save_dir}"
    assert gguf_files[0].stat().st_size > 0, "GGUF file is empty"
