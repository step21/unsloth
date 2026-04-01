import pytest
import torch

MPS = torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(not MPS, reason="MPS not available")


def test_cross_entropy_import():
    from unsloth.kernels import cross_entropy_loss
    assert hasattr(cross_entropy_loss, "_HAS_TRITON")
    assert cross_entropy_loss._HAS_TRITON is False


def test_rms_layernorm_import():
    from unsloth.kernels import rms_layernorm
    assert hasattr(rms_layernorm, "_HAS_TRITON")
    assert rms_layernorm._HAS_TRITON is False


def test_layernorm_import():
    from unsloth.kernels import layernorm
    assert hasattr(layernorm, "_HAS_TRITON")
    assert layernorm._HAS_TRITON is False


def test_rope_embedding_import():
    from unsloth.kernels import rope_embedding
    assert hasattr(rope_embedding, "fast_rope_embedding")


def test_swiglu_import():
    from unsloth.kernels import swiglu
    assert hasattr(swiglu, "_HAS_TRITON")
    assert swiglu._HAS_TRITON is False


def test_geglu_import():
    from unsloth.kernels import geglu
    assert hasattr(geglu, "_HAS_TRITON")
    assert geglu._HAS_TRITON is False


def test_fast_lora_import():
    from unsloth.kernels import fast_lora
    assert hasattr(fast_lora, "LoRA_MLP")
