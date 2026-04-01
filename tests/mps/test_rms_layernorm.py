import pytest
import torch

MPS = torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(not MPS, reason="MPS not available")

DEVICE = torch.device("mps")
DIM = 256
BATCH, SEQ = 2, 16


class _FakeRMSNorm(torch.nn.Module):
    """Minimal RMSNorm stand-in with the attributes fast_rms_layernorm reads."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps


def _reference_rms_layernorm(X, weight, eps):
    variance = X.float().pow(2).mean(-1, keepdim=True)
    normed = X.float() * torch.rsqrt(variance + eps)
    return (normed * weight.float()).to(X.dtype)


def test_fast_rms_layernorm_float32():
    from unsloth.kernels.rms_layernorm import fast_rms_layernorm
    ln = _FakeRMSNorm(DIM).to(DEVICE)
    X = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=torch.float32)
    out = fast_rms_layernorm(ln, X)
    ref = _reference_rms_layernorm(X, ln.weight, ln.variance_epsilon)
    assert out.device.type == "mps"
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_fast_rms_layernorm_float16():
    from unsloth.kernels.rms_layernorm import fast_rms_layernorm
    ln = _FakeRMSNorm(DIM).to(DEVICE).to(torch.float16)
    X = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=torch.float16)
    out = fast_rms_layernorm(ln, X)
    ref = _reference_rms_layernorm(X, ln.weight, ln.variance_epsilon)
    assert out.device.type == "mps"
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


def test_fast_rms_layernorm_gemma_mode():
    """gemma=True adds 1 to the weight before multiplying."""
    from unsloth.kernels.rms_layernorm import fast_rms_layernorm
    ln = _FakeRMSNorm(DIM).to(DEVICE)
    X = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=torch.float32)
    out = fast_rms_layernorm(ln, X, gemma=True)
    ref = _reference_rms_layernorm(X, ln.weight + 1.0, ln.variance_epsilon)
    assert out.device.type == "mps"
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
