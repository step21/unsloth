import pytest
import torch
import torch.nn.functional as F

MPS = torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(not MPS, reason="MPS not available")

DEVICE = torch.device("mps")
DIM = 256
BATCH, SEQ = 2, 16


def test_fast_layernorm_float32():
    from unsloth.kernels.layernorm import fast_layernorm
    ln = torch.nn.LayerNorm(DIM, eps=1e-5).to(DEVICE)
    X = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=torch.float32)
    out = fast_layernorm(ln, X)
    ref = F.layer_norm(X, (DIM,), weight=ln.weight, bias=ln.bias, eps=ln.eps)
    assert out.device.type == "mps"
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_fast_layernorm_float16():
    from unsloth.kernels.layernorm import fast_layernorm
    ln = torch.nn.LayerNorm(DIM, eps=1e-5).to(DEVICE).to(torch.float16)
    X = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=torch.float16)
    out = fast_layernorm(ln, X)
    ref = F.layer_norm(
        X.float(), (DIM,), weight=ln.weight.float(), bias=ln.bias.float(), eps=ln.eps
    ).to(torch.float16)
    assert out.device.type == "mps"
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
