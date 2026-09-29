import pytest

from semif_phase1.unsloth_backend import compare_predictions, left_pad, suggest_weights, throughput, weight_mode

from test_direct import Tokenizer

OPTIONS = [{"id": "yes", "description": "Yes."}, {"id": "no", "description": "No."}]


def rows(count):
    return [
        {"id": f"r{index}", "state": "evidence " * (index + 1), "question": f"Question {index}?", "options": OPTIONS}
        for index in range(count)
    ]


def test_left_pad_puts_every_answer_position_last():
    ids, masks = left_pad([[5, 6, 7], [8]], 0)
    assert ids == [[5, 6, 7], [0, 0, 8]]
    assert masks == [[1, 1, 1], [0, 0, 1]]
    with pytest.raises(ValueError):
        left_pad([[1], []], 0)


def test_weight_choices():
    assert weight_mode(False, False) == "16bit"
    assert weight_mode(True, False) == "bnb-4bit"
    with pytest.raises(ValueError):
        weight_mode(True, True)
    gib = 2**30
    assert suggest_weights(4_500_000_000, 24 * gib) == "16bit"
    assert suggest_weights(4_500_000_000, 8 * gib) == "bnb-4bit"
    with pytest.raises(ValueError):
        suggest_weights(30_000_000_000, 8 * gib)


def test_compare_predictions_aligns_by_option_id():
    reference = [
        {"id": "a", "option_ids": ["yes", "no"], "probabilities": [0.9, 0.1]},
        {"id": "b", "option_ids": ["yes", "no"], "probabilities": [0.4, 0.6]},
    ]
    candidate = [
        {"id": "a", "option_ids": ["no", "yes"], "probabilities": [0.2, 0.8]},
        {"id": "b", "option_ids": ["yes", "no"], "probabilities": [0.7, 0.3]},
    ]
    report = compare_predictions(reference, candidate)
    assert report["argmax_agreement"] == 0.5
    assert report["flips"][0]["id"] == "b"
    assert report["max_abs_probability_delta"] == pytest.approx(0.3)
    assert report["mean_total_variation"] == pytest.approx(0.2)


def test_throughput_summary():
    summary = throughput([{"input_tokens": 10, "allowed_token_mass": 0.9}, {"input_tokens": 30}], 2.0)
    assert summary["decisions_per_second"] == 1.0
    assert summary["mean_input_tokens"] == 20
    assert summary["min_allowed_token_mass"] == 0.9


class PaddingAwareModel:
    """Logits depend only on unmasked tokens, so padding must not move scores."""

    def __init__(self, torch, oom_above=None):
        self.torch = torch
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.oom_above = oom_above
        self.batches = []

    def parameters(self):
        yield self.weight

    def forward(self, input_ids, attention_mask, use_cache=False, return_dict=True, logits_to_keep=0):
        torch = self.torch
        if self.oom_above and input_ids.shape[0] > self.oom_above:
            raise torch.cuda.OutOfMemoryError("fake")
        self.batches.append(input_ids.shape[0])
        signal = ((input_ids * attention_mask).sum(-1, keepdim=True).float() + attention_mask.sum(-1, keepdim=True)) / 97.0
        vocab = torch.arange(256, dtype=torch.float32)
        last = torch.sin(signal * (vocab + 1))[:, None, :]
        return type("Output", (), {"logits": last})()

    __call__ = forward


def test_batched_scores_match_single_rows_and_back_off_on_oom():
    torch = pytest.importorskip("torch")
    from semif_phase1.direct import score as direct_score
    from semif_phase1.unsloth_backend import score_batched

    tokenizer = Tokenizer()
    tokenizer.pad_token_id, tokenizer.eos_token_id = 0, 0
    data = rows(7)
    model = PaddingAwareModel(torch, oom_above=2)
    batched = score_batched(model, tokenizer, data, {"source": "fake"}, batch_size=8)
    assert max(model.batches) == 2
    assert [row["id"] for row in batched] == [row["id"] for row in data]
    single = [direct_score(PaddingAwareModel(torch), tokenizer, row, {"source": "fake"}) for row in data]
    for left, right in zip(batched, single):
        assert left["probabilities"] == pytest.approx(right["probabilities"], abs=1e-6)
        assert left["prompt_sha256"] == right["prompt_sha256"]
        assert 0 < left["allowed_token_mass"] <= 1
    assert compare_predictions(single, batched)["argmax_agreement"] == 1.0
