import pytest

from gliner import GLiNER
from gliner.config import UniEncoderSpanRelexConfig
from gliner.data_processing import EventExtractionSpanProcessor, RelationExtractionSpanProcessor
from gliner.data_processing.collator import EventExtractionSpanDataCollator, RelationExtractionSpanDataCollator
from gliner.decoding.decoder import EventSpanDecoder, SpanRelexDecoder
from gliner.model import BaseEncoderGLiNER, UniEncoderSpanRelexGLiNER


class _WordsSplitter:
    def __call__(self, text):
        cursor = 0
        for token in text.split():
            start = text.index(token, cursor)
            end = start + len(token)
            cursor = end
            yield token, start, end


def _minimal_encoder_model(cls=BaseEncoderGLiNER):
    model = cls.__new__(cls)
    model.data_processor = type("Processor", (), {"words_splitter": _WordsSplitter()})()
    return model


def test_span_model():
    model = GLiNER.from_pretrained("gliner-community/gliner_small-v2.5")

    text = """
    Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
    """

    labels = ["person", "award", "date", "competitions", "teams", "person"]

    entities = model.predict_entities(text, labels)

    assert len(entities) > 0


def test_prepare_batch_aligns_per_text_labels_after_empty_text_filtering():
    model = _minimal_encoder_model()

    prepared = model.prepare_batch(
        ["", "John works at Acme"],
        [["wrong"], ["person", "organization"]],
    )

    assert prepared["valid_texts"] == ["John works at Acme"]
    assert prepared["valid_to_orig_idx"] == [1]
    assert prepared["entity_types"] == [["person", "organization"]]


def test_prepare_batch_validates_per_text_label_count():
    model = _minimal_encoder_model()

    with pytest.raises(ValueError, match="Per-text labels must have length 2"):
        model.prepare_batch(["one", "two"], [["label"]])


def test_relex_prepare_batch_aligns_per_text_relations_after_empty_text_filtering():
    model = _minimal_encoder_model(UniEncoderSpanRelexGLiNER)

    prepared = model.prepare_batch(
        ["", "John works at Acme"],
        [["wrong"], ["person", "organization"]],
        relations=[["wrong_relation"], ["works_at"]],
    )

    assert prepared["valid_texts"] == ["John works at Acme"]
    assert prepared["entity_types"] == [["person", "organization"]]
    assert prepared["relation_types"] == [["works_at"]]


def test_relex_prepare_batch_validates_per_text_relation_count():
    model = _minimal_encoder_model(UniEncoderSpanRelexGLiNER)

    with pytest.raises(ValueError, match="Per-text relations must have length 2"):
        model.prepare_batch(
            ["one", "two"],
            [["person"], ["organization"]],
            relations=[["works_at"]],
        )


def _relex_model_with_config(event_mode, trigger_types=None):
    model = UniEncoderSpanRelexGLiNER.__new__(UniEncoderSpanRelexGLiNER)
    model.config = UniEncoderSpanRelexConfig(
        model_name="prajjwal1/bert-tiny",
        relations_layer="dot",
        event_mode=event_mode,
        trigger_types=trigger_types,
    )
    return model


def test_event_forward_exposes_all_loss_components():
    """On a real event batch the forward exposes span/adj/rel raw losses
    (detached) so training can log them per component."""
    import torch
    from gliner.data_processing.trigger_types import derive_trigger_types

    recs = [{
        "tokenized_text": ["John", "attacked", "Paris"],
        "ner": [[0, 0, "Person"], [1, 1, "Attack"], [2, 2, "Location"]],
        "relations": [[1, 0, "Attacker"], [1, 2, "Place"]],
    }]
    trig = sorted(derive_trigger_types(recs))
    cfg = UniEncoderSpanRelexConfig(
        model_name="prajjwal1/bert-tiny", relations_layer="dot", event_mode=True,
        trigger_types=trig, max_width=4, max_len=64,
    )
    model = UniEncoderSpanRelexGLiNER.load_from_config(cfg)
    collator = model.data_collator_class(cfg, data_processor=model.data_processor, prepare_labels=True)
    batch = collator(recs, entity_types=[["Attack", "Person", "Location"]],
                     relation_types=[["Attacker", "Place"]])
    tensors = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
    tensors["text_lengths"] = batch["text_lengths"]

    model.train()
    out = model.model(**tensors)
    for comp in (out.span_loss, out.adj_loss, out.rel_loss):
        assert comp is not None and torch.isfinite(comp) and not comp.requires_grad


class TestEventModeClassDispatch:
    def test_event_mode_selects_event_classes(self):
        model = _relex_model_with_config(event_mode=True, trigger_types=["Life.Die"])

        assert model.data_processor_class is EventExtractionSpanProcessor
        assert model.data_collator_class is EventExtractionSpanDataCollator
        assert model.decoder_class is EventSpanDecoder

    def test_non_event_mode_selects_relex_classes(self):
        model = _relex_model_with_config(event_mode=False)

        assert model.data_processor_class is RelationExtractionSpanProcessor
        assert model.data_collator_class is RelationExtractionSpanDataCollator
        assert model.decoder_class is SpanRelexDecoder
