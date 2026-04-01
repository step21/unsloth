"""
Tests for SwiGLU kernels on MPS.

Covers:
  - Forward output matches PyTorch reference
  - Backward (swiglu_DWf_DW_dfg_kernel) produces correct dtype on float16 inputs,
    guarding against the bug where 'e = e.to(float32)' would rebind the local
    variable instead of using a temp, causing a dtype mismatch in fast_lora.py backward.
"""

import pytest
import torch

MPS = torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(not MPS, reason="MPS not available")

DEVICE = torch.device("mps")
BATCH, SEQ, DIM = 2, 8, 64


def _reference_swiglu_forward(e, g):
    return e * torch.sigmoid(e.float()).to(e.dtype) * g


def _reference_swiglu_backward(DW, e, g):
    """Returns (h, df, de) matching the in-place kernel outputs."""
    orig_dtype = e.dtype
    e_f = e.float()
    se = torch.sigmoid(e_f)
    f = (se * e_f).to(orig_dtype)
    h = f * g
    df = DW * f
    dg = DW * g
    de = (dg.float() * se * (1.0 + e_f * (1.0 - se))).to(orig_dtype)
    return h, df, de


def test_swiglu_forward_float32():
    from unsloth.kernels.swiglu import swiglu_fg_kernel
    e = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=torch.float32)
    g = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=torch.float32)
    out = swiglu_fg_kernel(e, g)
    ref = _reference_swiglu_forward(e, g)
    assert out.device.type == "mps"
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_swiglu_forward_float16():
    from unsloth.kernels.swiglu import swiglu_fg_kernel
    e = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=torch.float16)
    g = torch.randn(BATCH, SEQ, DIM, device=DEVICE, dtype=torch.float16)
    out = swiglu_fg_kernel(e, g)
    ref = _reference_swiglu_forward(e, g)
    assert out.device.type == "mps"
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


def test_swiglu_backward_float32_dtype_preserved():
    """swiglu_DWf_DW_dfg_kernel does not change dtypes on float32 inputs."""
    from unsloth.kernels.swiglu import swiglu_DWf_DW_dfg_kernel
    DW = torch.randn(BATCH * SEQ, DIM, device=DEVICE, dtype=torch.float32)
    e = torch.randn(BATCH * SEQ, DIM, device=DEVICE, dtype=torch.float32)
    g = torch.randn(BATCH * SEQ, DIM, device=DEVICE, dtype=torch.float32)
    DW_out, e_out, g_out = swiglu_DWf_DW_dfg_kernel(DW, e, g)
    assert DW_out.dtype == torch.float32
    assert e_out.dtype == torch.float32
    assert g_out.dtype == torch.float32


def test_swiglu_backward_float16_dtype_preserved():
    """swiglu_DWf_DW_dfg_kernel outputs stay float16 when inputs are float16."""
    from unsloth.kernels.swiglu import swiglu_DWf_DW_dfg_kernel
    DW = torch.randn(BATCH * SEQ, DIM, device=DEVICE, dtype=torch.float16)
    e = torch.randn(BATCH * SEQ, DIM, device=DEVICE, dtype=torch.float16)
    g = torch.randn(BATCH * SEQ, DIM, device=DEVICE, dtype=torch.float16)
    DW_out, e_out, g_out = swiglu_DWf_DW_dfg_kernel(DW, e, g)
    assert DW_out.dtype == torch.float16, f"DW dtype: {DW_out.dtype}"
    assert e_out.dtype == torch.float16, f"e dtype: {e_out.dtype}"
    assert g_out.dtype == torch.float16, f"g dtype: {g_out.dtype}"


def test_swiglu_backward_float16_values():
    """swiglu_DWf_DW_dfg_kernel values match reference on float16 inputs."""
    from unsloth.kernels.swiglu import swiglu_DWf_DW_dfg_kernel
    DW = torch.randn(BATCH * SEQ, DIM, device=DEVICE, dtype=torch.float16)
    e = torch.randn(BATCH * SEQ, DIM, device=DEVICE, dtype=torch.float16)
    g = torch.randn(BATCH * SEQ, DIM, device=DEVICE, dtype=torch.float16)
    DW_c, e_c, g_c = DW.clone(), e.clone(), g.clone()
    swiglu_DWf_DW_dfg_kernel(DW, e, g)
    h_ref, df_ref, de_ref = _reference_swiglu_backward(DW_c, e_c, g_c)
    torch.testing.assert_close(DW.float(), h_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(e.float(), df_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(g.float(), de_ref.float(), atol=1e-2, rtol=1e-2)
