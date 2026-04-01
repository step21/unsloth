"""
Tests for LoRA_MLP forward/backward on MPS.

Checks that gradients for d_upB have the correct dtype — guarding against
the scenario where intermediate float32 casts in the backward kernel
produce float32 gradient tensors when float16 weights are expected.
"""

import pytest
import torch

MPS = torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(not MPS, reason="MPS not available")

DEVICE = torch.device("mps")
HIDDEN = 64
INTERMEDIATE = 128
RANK = 8
BATCH, SEQ = 2, 4


def _make_lora_weights(in_dim, out_dim, rank, dtype):
    W = torch.randn(out_dim, in_dim, device=DEVICE, dtype=dtype)    # (out, in): matmul_lora uses W.t()
    A = torch.randn(rank, in_dim, device=DEVICE, dtype=dtype)        # (rank, in): matmul_lora does A.t() → (in, rank)
    B = torch.zeros(out_dim, rank, device=DEVICE, dtype=dtype)       # (out, rank): matmul_lora does B.t() → (rank, out)
    scale = 1.0
    return W, None, A, B, scale


def test_lora_mlp_forward_backward_float32():
    """LoRA_MLP forward + backward runs without error on MPS in float32."""
    from unsloth.kernels.fast_lora import LoRA_MLP
    from unsloth.kernels.swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel

    dtype = torch.float32
    X = torch.randn(BATCH, SEQ, HIDDEN, device=DEVICE, dtype=dtype, requires_grad=True)
    gateW, gateWq, gateA, gateB, gateS = _make_lora_weights(HIDDEN, INTERMEDIATE, RANK, dtype)
    upW, upWq, upA, upB, upS = _make_lora_weights(HIDDEN, INTERMEDIATE, RANK, dtype)
    downW, downWq, downA, downB, downS = _make_lora_weights(INTERMEDIATE, HIDDEN, RANK, dtype)
    for t in [gateA, gateB, upA, upB, downA, downB]:
        t.requires_grad_(True)

    out = LoRA_MLP.apply(
        X,
        gateW, gateWq, gateA, gateB, gateS,
        upW, upWq, upA, upB, upS,
        downW, downWq, downA, downB, downS,
        swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel,
    )
    loss = out.sum()
    loss.backward()

    assert upB.grad is not None, "upB gradient is None"
    assert upB.grad.dtype == dtype, f"upB.grad dtype: {upB.grad.dtype}, expected {dtype}"
    assert upB.grad.device.type == "mps"


def test_lora_mlp_forward_backward_float16():
    """LoRA_MLP d_upB gradient has correct float16 dtype on MPS."""
    from unsloth.kernels.fast_lora import LoRA_MLP
    from unsloth.kernels.swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel

    dtype = torch.float16
    X = torch.randn(BATCH, SEQ, HIDDEN, device=DEVICE, dtype=dtype, requires_grad=True)
    gateW, gateWq, gateA, gateB, gateS = _make_lora_weights(HIDDEN, INTERMEDIATE, RANK, dtype)
    upW, upWq, upA, upB, upS = _make_lora_weights(HIDDEN, INTERMEDIATE, RANK, dtype)
    downW, downWq, downA, downB, downS = _make_lora_weights(INTERMEDIATE, HIDDEN, RANK, dtype)
    for t in [gateA, gateB, upA, upB, downA, downB]:
        t.requires_grad_(True)

    out = LoRA_MLP.apply(
        X,
        gateW, gateWq, gateA, gateB, gateS,
        upW, upWq, upA, upB, upS,
        downW, downWq, downA, downB, downS,
        swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel,
    )
    loss = out.sum()
    loss.backward()

    assert upB.grad is not None, "upB gradient is None"
    assert upB.grad.dtype == dtype, (
        f"upB.grad dtype: {upB.grad.dtype}, expected {dtype} — "
        "dtype mismatch indicates float32 cast was not converted back"
    )
    assert upB.grad.device.type == "mps"
