"""Sanity/overfit training check for the event architecture.

This is not a benchmark and does not claim SOTA-quality convergence -- this
machine has no CUDA (macOS, MPS/CPU only, see RE_DIFFERENCES-adjacent
findings reported earlier in this work), so "real training run" here means:
does the loss actually go down when the model trains on real (trigger,
role, argument) supervision, on real WikiEvents text? That's a materially
different, stronger claim than "forward/backward doesn't crash" (already
covered by test_wikievents_vertical_slice.py) -- a wiring bug that produces
a technically-finite but structurally-wrong loss (e.g. misaligned
adj_matrix/rel_matrix, discussed and specifically tested for in
test_data_processing.py's TestEventExtractionSpanProcessor) would still let
loss decrease on the *classification* term alone while the event-specific
terms stay flat. So this checks adj_loss and rel_loss individually, not
just total loss.

An earlier version of this test used ``triples_layer="TransE"`` and only
asserted on total loss, which passed while rel_loss was completely dead:
TransE's default clamp_norm=10.0 saturates the translational distance
``||h + r - t||`` for essentially any triple at standard transformer
init scale (empirically, mean L1 distance ~88.66 at hidden_size=64 --
100% of 1000 sampled triples exceeded the clamp), and torch.clamp's
gradient is exactly zero past its ceiling, so no signal reached the
relation-scoring path at all. This test now uses ``triples_layer=None``
(concatenation + dot-product against relation-prompt embeddings), which
has no clamp. See docs/events.md for the full writeup.

This test is a wiring check, not a discrimination or output-quality
benchmark: it proves gradient reaches and meaningfully moves adj_loss and
rel_loss (catching a dead path like TransE's), not that the model learns
to output correct predictions. Role-classification quality at production
scale is unverified -- this CI-scale harness (hidden_size=64, one short
document, 15 steps) cannot validate learning quality for any
classification head in this model. A real training run is needed to
answer that question.
"""

import json
import sys
from pathlib import Path

import torch
import pytest
from transformers import AutoTokenizer

from gliner.config import UniEncoderSpanRelexConfig
from gliner.modeling.base import UniEncoderSpanRelexModel
from gliner.data_processing import EventExtractionSpanProcessor, EventExtractionSpanDataCollator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from _trigger_types import derive_trigger_types  # noqa: E402

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "wikievents_sample.jsonl"


def _load_small_records(max_tokens=200):
    records = []
    with FIXTURE_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if len(rec["tokenized_text"]) > max_tokens:
                continue
            records.append({
                "tokenized_text": rec["tokenized_text"],
                "ner": [tuple(e) for e in rec["ner"]],
                "relations": [tuple(r) for r in rec["relations"]],
            })
    return records


class TestEventTrainingSanity:
    def test_loss_decreases_over_a_short_training_run(self):
        records = _load_small_records()
        assert records, "expected at least one small fixture record"

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_tokens(["<<ENT>>", "<<REL>>"], special_tokens=True)

        # Derived from relation-head labels, not a label-shape heuristic --
        # see data/_trigger_types.py.
        trigger_types = derive_trigger_types(records)

        config = UniEncoderSpanRelexConfig(
            model_name="bert-base-uncased",
            hidden_size=64,
            dropout=0.1,
            max_width=8,
            max_len=384,
            span_mode="markerV0",
            class_token_index=tokenizer.convert_tokens_to_ids("<<ENT>>"),
            rel_token_index=tokenizer.convert_tokens_to_ids("<<REL>>"),
            vocab_size=len(tokenizer),
            has_rnn=False,
            post_fusion_schema="",
            embed_ent_token=True,
            embed_rel_token=True,
            relations_layer="dot",
            triples_layer=None,
            event_mode=True,
            augment_data_prob=0.0,
            max_types=30,
        )

        processor = EventExtractionSpanProcessor(config, tokenizer, words_splitter=None, trigger_types=trigger_types)
        collator = EventExtractionSpanDataCollator(config, data_processor=processor, prepare_labels=True)

        model = UniEncoderSpanRelexModel(config, from_pretrained=False)
        model.token_rep_layer.resize_token_embeddings(len(tokenizer))
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

        # Capture adj_loss (trigger/argument span selection) and rel_loss
        # (argument-role classification) individually, not just the total --
        # see module docstring for why total-loss-only was insufficient.
        adj_losses = []
        rel_losses = []
        orig_adj_loss = model.adj_loss
        orig_rel_loss = model.rel_loss

        def spy_adj_loss(*args, **kwargs):
            value = orig_adj_loss(*args, **kwargs)
            adj_losses.append(value.item())
            return value

        def spy_rel_loss(*args, **kwargs):
            value = orig_rel_loss(*args, **kwargs)
            rel_losses.append(value.item())
            return value

        model.adj_loss = spy_adj_loss
        model.rel_loss = spy_rel_loss

        total_losses = []
        num_steps = 15
        for _ in range(num_steps):
            batch = collator(records)  # re-collate each step: negative sampling varies, matching real training
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            total_losses.append(output.loss.item())

        # First-vs-last average over a small window smooths batch-to-batch
        # noise from negative resampling without hiding a real trend.
        early = sum(total_losses[:3]) / 3
        late = sum(total_losses[-3:]) / 3
        assert late < early, f"loss did not decrease: early={early:.4f} late={late:.4f} (all={total_losses})"
        assert late < early * 0.9, (
            f"loss decreased too little to call this a real training signal: "
            f"early={early:.4f} late={late:.4f} (all={total_losses})"
        )

        adj_early = sum(adj_losses[:3]) / 3
        adj_late = sum(adj_losses[-3:]) / 3
        assert adj_late < adj_early, (
            f"adj_loss (trigger/argument selection) did not decrease: "
            f"early={adj_early:.4f} late={adj_late:.4f} (all={adj_losses})"
        )

        rel_early = sum(rel_losses[:3]) / 3
        rel_late = sum(rel_losses[-3:]) / 3
        assert rel_late < rel_early * 0.7, (
            f"rel_loss (argument-role classification) did not show real "
            f"training signal -- this is the component that was silently "
            f"dead under triples_layer='TransE' (see module docstring): "
            f"early={rel_early:.4f} late={rel_late:.4f} (all={rel_losses})"
        )
