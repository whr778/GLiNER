"""Regression tests for two dead-config-knob bugs in gliner/modeling/base.py.

Both bugs share a shape: a loss-shaping kwarg reaches a ``.loss()`` method
(every real call is kwargs-only, from ``forward()`` -- see below), but the
method silently fails to route it into ``BaseModel._loss()``, so the config
value has zero effect on training.

1. ``masking`` (negative-sampling strategy) was absorbed into an "unused"
   ``**kwargs`` catch-all for every token-level model's ``.loss()``, instead
   of being forwarded to ``_loss()``.
2. ``prob_margin`` was missing from two ``.loss()`` signatures entirely
   (``BiEncoderSpanModel``, ``UniEncoderSpanDecoderModel``); the positional
   call into ``_loss()`` then shifted ``label_smoothing`` into ``_loss()``'s
   ``prob_margin`` slot, so real label smoothing was always inert.

These methods only touch ``self._loss`` (a stateless call into
``focal_loss_with_logits``), so a minimal stub with ``_loss`` bound is enough
-- no full model/config/encoder needed.
"""

from types import SimpleNamespace

import torch

from gliner.modeling.base import (
    BaseModel,
    BiEncoderSpanModel,
    UniEncoderSpanDecoderModel,
    UniEncoderTokenModel,
    UniEncoderTokenRelexModel,
)


def _stub():
    s = SimpleNamespace()
    s._loss = BaseModel._loss.__get__(s)
    return s


def _token_inputs():
    torch.manual_seed(0)
    B, W, C = 2, 4, 3
    scores = torch.randn(B, W, C, 3)
    labels = torch.zeros(B, W, C, 3)
    labels[0, 1, 0] = 1.0
    labels[1, 2, 1] = 1.0
    prompts_embedding_mask = torch.ones(B, C)
    word_mask = torch.ones(B, W)
    return scores, labels, prompts_embedding_mask, word_mask


def _span_inputs():
    torch.manual_seed(0)
    B, L, K, C = 2, 3, 2, 3
    scores = torch.randn(B, L, K, C)
    labels = torch.zeros(B, L, K, C)
    labels[0, 0, 0, 0] = 1.0
    labels[1, 1, 1, 1] = 1.0
    prompts_embedding_mask = torch.ones(B, C)
    span_mask = torch.ones(B, L, K)
    return scores, labels, prompts_embedding_mask, span_mask


class TestMaskingWiredForTokenModels:
    """masking='global', negatives=0.0 zeroes every negative-cell loss --
    if masking is dropped (the bug), the result is unchanged from masking='none'.
    """

    def test_uni_encoder_token_model_masking_applied(self):
        scores, labels, pem, word_mask = _token_inputs()
        stub = _stub()

        unmasked = UniEncoderTokenModel.loss(stub, scores, labels, pem, word_mask, masking="none")
        masked = UniEncoderTokenModel.loss(
            stub, scores, labels, pem, word_mask, masking="global", negatives=0.0
        )

        assert not torch.allclose(unmasked, masked)
        assert masked.item() < unmasked.item()

    def test_uni_encoder_token_relex_model_masking_applied(self):
        scores, labels, pem, word_mask = _token_inputs()
        stub = _stub()

        unmasked = UniEncoderTokenRelexModel.loss(stub, scores, labels, pem, word_mask, masking="none")
        masked = UniEncoderTokenRelexModel.loss(
            stub, scores, labels, pem, word_mask, masking="global", negatives=0.0
        )

        assert not torch.allclose(unmasked, masked)
        assert masked.item() < unmasked.item()

    def test_label_and_span_masking_do_not_crash_on_4d_token_labels(self):
        """"label"/"span" masking were written for 3D span labels (B, N, C);
        token labels are 4D (B, W, C, 3). No shipped config reaches this
        combination (all set masking: 'none'), so this was unreachable before
        the masking fix and is unverified beyond "doesn't crash" -- whether
        summing over W vs. C is the intended semantics for token models is
        not asserted here.
        """
        scores, labels, pem, word_mask = _token_inputs()
        stub = _stub()

        for mode in ("label", "span"):
            loss = UniEncoderTokenModel.loss(stub, scores, labels, pem, word_mask, masking=mode)
            assert torch.isfinite(loss).all()


class TestProbMarginNotConflatedWithLabelSmoothing:
    """prob_margin and label_smoothing must independently reach _loss(), not
    collide into a single positional slot.
    """

    def test_bi_encoder_span_model_prob_margin_has_effect(self):
        scores, labels, pem, mask_label = _span_inputs()
        stub = _stub()

        baseline = BiEncoderSpanModel.loss(stub, scores, labels, pem, mask_label, prob_margin=0.0)
        with_margin = BiEncoderSpanModel.loss(stub, scores, labels, pem, mask_label, prob_margin=0.5)

        assert not torch.allclose(baseline, with_margin), "prob_margin=0.5 had no effect -- still dead"

    def test_bi_encoder_span_model_matches_direct_loss_call(self):
        scores, labels, pem, mask_label = _span_inputs()
        stub = _stub()

        via_method = BiEncoderSpanModel.loss(
            stub, scores, labels, pem, mask_label, prob_margin=0.3, label_smoothing=0.1, reduction="sum"
        )

        BS, _, _, CL = scores.shape
        num_classes = pem.shape[-1]
        flat_scores = scores.view(BS, -1, CL)
        flat_labels = labels.view(BS, -1, CL)
        expected = stub._loss(flat_scores, flat_labels, -1.0, 0.0, 0.3, 0.1, negatives=1.0, masking="none")
        expected = (expected.view(BS, -1, num_classes) * pem.unsqueeze(1)).sum()

        assert torch.allclose(via_method, expected)

    def test_uni_encoder_span_decoder_model_prob_margin_has_effect(self):
        scores, labels, pem, mask_label = _span_inputs()
        stub = _stub()

        baseline = UniEncoderSpanDecoderModel.loss(stub, scores, labels, pem, mask_label, prob_margin=0.0)
        with_margin = UniEncoderSpanDecoderModel.loss(stub, scores, labels, pem, mask_label, prob_margin=0.5)

        assert not torch.allclose(baseline, with_margin), "prob_margin=0.5 had no effect -- still dead"

    def test_uni_encoder_span_decoder_model_matches_direct_loss_call(self):
        """Both params must reach _loss() under their own name, not swapped --
        a weaker "these two configs give different results" check would pass
        even on the buggy code, since a mislabeled value still changes the
        output; only an exact-match against a correctly-attributed manual
        call actually discriminates.
        """
        scores, labels, pem, mask_label = _span_inputs()
        stub = _stub()

        via_method = UniEncoderSpanDecoderModel.loss(
            stub, scores, labels, pem, mask_label, prob_margin=0.3, label_smoothing=0.1, reduction="sum"
        )

        BS, _, _, CL = scores.shape
        num_classes = pem.shape[-1]
        flat_scores = scores.view(BS, -1, CL)
        flat_labels = labels.view(BS, -1, CL)
        expected = stub._loss(flat_scores, flat_labels, -1.0, 0.0, 0.3, 0.1, negatives=1.0, masking="none")
        expected = (expected.view(BS, -1, num_classes) * pem.unsqueeze(1)).sum()

        assert torch.allclose(via_method, expected)
