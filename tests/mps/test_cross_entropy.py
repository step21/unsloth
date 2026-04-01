import pytest
import torch
import torch.nn.functional as F

MPS = torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(not MPS, reason="MPS not available")

DEVICE = torch.device("mps")
VOCAB = 256
BATCH, SEQ = 2, 8


def _make_inputs(dtype=torch.float32):
    logits = torch.randn(BATCH, SEQ, VOCAB, device=DEVICE, dtype=dtype)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    labels[0, 3] = -100
    return logits, labels


def _reference_ce(logits, labels):
    batch, seq, vocab = logits.shape
    return F.cross_entropy(
        logits.float().view(batch * seq, vocab),
        labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    ) / (labels != -100).sum()


def test_fast_cross_entropy_matches_reference_float32():
    from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss
    logits, labels = _make_inputs(torch.float32)
    n_items = (labels != -100).sum()
    out = fast_cross_entropy_loss(logits, labels, n_items=n_items)
    ref = _reference_ce(logits, labels)
    assert out.device.type == "mps"
    torch.testing.assert_close(out.cpu().float(), ref.cpu().float(), atol=1e-4, rtol=1e-4)


def test_fast_cross_entropy_matches_reference_float16():
    from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss
    logits, labels = _make_inputs(torch.float16)
    n_items = (labels != -100).sum()
    out = fast_cross_entropy_loss(logits, labels, n_items=n_items)
    ref = _reference_ce(logits, labels)
    assert out.device.type == "mps"
    torch.testing.assert_close(out.cpu().float(), ref.cpu().float(), atol=1e-2, rtol=1e-2)


def test_fast_cross_entropy_with_softcapping():
    from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss
    logits, labels = _make_inputs(torch.float32)
    n_items = (labels != -100).sum()
    out = fast_cross_entropy_loss(logits, labels, logit_softcapping=30.0, n_items=n_items)
    assert torch.isfinite(out)
    assert out.device.type == "mps"
