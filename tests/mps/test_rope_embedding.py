"""
Tests for RoPE embedding MPS device-correctness.

Covers the set of bugs fixed in this session:
  - device.index or 0  (TypeError when device.index is None on MPS)
  - get_cached with None device_index raises TypeError
  - loader.py re-initialization overwriting MPS cache with CPU tensor
  - position_embeddings branch missing .to(device, dtype)
"""

import pytest
import torch

MPS = torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(not MPS, reason="MPS not available")

DEVICE = torch.device("mps")
DIM = 64
MAX_POS = 128
SEQ = 16


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _reference_rope(Q, K, cos, sin):
    return (Q * cos) + (_rotate_half(Q) * sin), (K * cos) + (_rotate_half(K) * sin)


def test_fast_rope_embedding_output_on_mps():
    """fast_rope_embedding returns tensors on mps:0 and matches reference."""
    from unsloth.kernels.rope_embedding import fast_rope_embedding
    batch, heads, seq = 1, 4, SEQ
    Q = torch.randn(batch, heads, seq, DIM, device=DEVICE, dtype=torch.float32)
    K = torch.randn(batch, heads, seq, DIM, device=DEVICE, dtype=torch.float32)
    cos = torch.randn(seq, DIM, device=DEVICE, dtype=torch.float32)
    sin = torch.randn(seq, DIM, device=DEVICE, dtype=torch.float32)
    Q_out, K_out = fast_rope_embedding(Q, K, cos, sin)
    Q_ref, K_ref = _reference_rope(Q, K, cos, sin)
    assert Q_out.device.type == "mps"
    assert K_out.device.type == "mps"
    torch.testing.assert_close(Q_out, Q_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(K_out, K_ref, atol=1e-4, rtol=1e-4)


def test_llama_rotary_embedding_cos_sin_on_mps():
    """LlamaRotaryEmbedding cos/sin cache stays on mps:0 after __init__."""
    from unsloth.models.llama import LlamaRotaryEmbedding
    emb = LlamaRotaryEmbedding(dim=DIM, max_position_embeddings=MAX_POS, device=DEVICE)
    x = torch.zeros(1, 1, SEQ, DIM, device=DEVICE, dtype=torch.float32)
    cos, sin = emb(x, seq_len=SEQ)
    assert cos.device.type == "mps", f"cos on {cos.device}, expected mps"
    assert sin.device.type == "mps", f"sin on {sin.device}, expected mps"


def test_llama_get_cached_none_device_index_no_type_error():
    """get_cached(seq_len, device_index=None) must not raise TypeError on MPS."""
    from unsloth.models.llama import LlamaRotaryEmbedding
    emb = LlamaRotaryEmbedding(dim=DIM, max_position_embeddings=MAX_POS, device=DEVICE)
    x = torch.zeros(1, 1, SEQ, DIM, device=DEVICE, dtype=torch.float32)
    emb(x, seq_len=SEQ)
    cos, sin = emb.get_cached(SEQ, device_index=None)
    assert cos is not None
    assert sin is not None
    assert cos.device.type == "mps"


def test_llama_set_cos_sin_cache_reinit_stays_on_mps():
    """
    Simulates the loader.py re-initialization path:
    _set_cos_sin_cache is called with torch.device(DEVICE_TYPE_TORCH).
    The cache must remain on mps:0, not fall back to CPU.
    """
    from unsloth.models.llama import LlamaRotaryEmbedding
    from unsloth.device_type import DEVICE_TYPE_TORCH
    emb = LlamaRotaryEmbedding(dim=DIM, max_position_embeddings=MAX_POS, device=DEVICE)
    reinit_device = torch.device(DEVICE_TYPE_TORCH)
    emb._set_cos_sin_cache(MAX_POS, reinit_device, torch.float32)
    cos = emb.multi_gpu_cos_cached[0]
    sin = emb.multi_gpu_sin_cached[0]
    assert cos.device.type == "mps", f"cos on {cos.device} after re-init, expected mps"
    assert sin.device.type == "mps", f"sin on {sin.device} after re-init, expected mps"


def test_gemma_rotary_embedding_cos_sin_on_mps():
    """GemmaFixedRotaryEmbedding cos/sin cache stays on mps:0."""
    from unsloth.models.gemma import GemmaFixedRotaryEmbedding
    emb = GemmaFixedRotaryEmbedding(dim=DIM, max_position_embeddings=MAX_POS, device=DEVICE)
    x = torch.zeros(1, 1, SEQ, DIM, device=DEVICE, dtype=torch.float32)
    cos, sin = emb(x, seq_len=SEQ)
    assert cos.device.type == "mps", f"cos on {cos.device}, expected mps"
    assert sin.device.type == "mps", f"sin on {sin.device}, expected mps"


def test_gemma_get_cached_none_device_index_no_type_error():
    """GemmaFixedRotaryEmbedding.get_cached with None device_index must not raise."""
    from unsloth.models.gemma import GemmaFixedRotaryEmbedding
    emb = GemmaFixedRotaryEmbedding(dim=DIM, max_position_embeddings=MAX_POS, device=DEVICE)
    x = torch.zeros(1, 1, SEQ, DIM, device=DEVICE, dtype=torch.float32)
    emb(x, seq_len=SEQ)
    cos, sin = emb.get_cached(SEQ, device_index=None)
    assert cos is not None
    assert cos.device.type == "mps"
