"""
Tests for GEGLU kernels on MPS.

Covers:
  - Forward (exact and approx) output matches PyTorch reference
  - Backward kernels produce correct dtype on float16 inputs.

Note: geglu_exact_backward_kernel and geglu_approx_backward_kernel contain a
known bug where 'e = e.to(float32)' rebinds the local variable, so on float16
inputs e.copy_(df) writes into the float32 copy rather than the original tensor.
The dtype-preservation tests here will surface that bug if it is still present.
"""

import pytest
import torch
import torch.nn.functional as F

MPS = torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(not MPS, reason="MPS not available")

DEVICE = torch.device("mps")
BATCH, SEQ, DIM = 2, 8, 64


def test_geglu_exact_forward_float32():
    from unsloth.kernels.geglu import geglu_exact_forward_kernel
    gate = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=torch.float32)
    up = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=torch.float32)
    out = geglu_exact_forward_kernel(gate, up)
    ref = F.gelu(gate, approximate="none") * up
    assert out.device.type == "mps"
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_geglu_exact_forward_float16():
    from unsloth.kernels.geglu import geglu_exact_forward_kernel
    gate = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=torch.float16)
    up = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=torch.float16)
    out = geglu_exact_forward_kernel(gate, up)
    ref = F.gelu(gate.float(), approximate="none").to(torch.float16) * up
    assert out.device.type == "mps"
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


def test_geglu_approx_forward_float32():
    from unsloth.kernels.geglu import geglu_approx_forward_kernel
    gate = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=torch.float32)
    up = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=torch.float32)
    out = geglu_approx_forward_kernel(gate, up)
    ref = F.gelu(gate, approximate="tanh") * up
    assert out.device.type == "mps"
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_geglu_approx_forward_float16():
    from unsloth.kernels.geglu import geglu_approx_forward_kernel
    gate = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=torch.float16)
    up = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=torch.float16)
    out = geglu_approx_forward_kernel(gate, up)
    ref = F.gelu(gate.float(), approximate="tanh").to(torch.float16) * up
    assert out.device.type == "mps"
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


def test_geglu_exact_backward_float16_dtype_preserved():
    """geglu_exact_backward_kernel outputs stay float16 when inputs are float16."""
    from unsloth.kernels.geglu import geglu_exact_backward_kernel
    DW = torch.randn(BATCH * SEQ, DIM, device=DEVICE, dtype=torch.float16)
    e = torch.randn(BATCH * SEQ, DIM, device=DEVICE, dtype=torch.float16)
    g = torch.randn(BATCH * SEQ, DIM, device=DEVICE, dtype=torch.float16)
    DW_out, e_out, g_out = geglu_exact_backward_kernel(DW, e, g)
    assert DW_out.dtype == torch.float16, f"DW dtype: {DW_out.dtype}"
    assert e_out.dtype == torch.float16, f"e dtype: {e_out.dtype}"
    assert g_out.dtype == torch.float16, f"g dtype: {g_out.dtype}"


def test_geglu_approx_backward_float16_dtype_preserved():
    """geglu_approx_backward_kernel outputs stay float16 when inputs are float16."""
    from unsloth.kernels.geglu import geglu_approx_backward_kernel
    DW = torch.randn(BATCH * SEQ, DIM, device=DEVICE, dtype=torch.float16)
    e = torch.randn(BATCH * SEQ, DIM, device=DEVICE, dtype=torch.float16)
    g = torch.randn(BATCH * SEQ, DIM, device=DEVICE, dtype=torch.float16)
    DW_out, e_out, g_out = geglu_approx_backward_kernel(DW, e, g)
    assert DW_out.dtype == torch.float16, f"DW dtype: {DW_out.dtype}"
    assert e_out.dtype == torch.float16, f"e dtype: {e_out.dtype}"
    assert g_out.dtype == torch.float16, f"g dtype: {g_out.dtype}"
