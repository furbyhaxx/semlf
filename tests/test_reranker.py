import pytest

from semif_phase1.reranker import _answer_ids, _encode


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode())


class AnswerTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"no": [4], "yes": [7]}[text]

    def convert_tokens_to_ids(self, text):
        return {"no": 4, "yes": 7}[text]


def test_reranker_pair_contains_only_declared_semantics():
    row = {
        "id": "case",
        "state": "STATE",
        "question": "QUESTION",
        "options": [{"id": "a", "description": "ANSWER"}, {"id": "b", "description": "OTHER"}],
        "label": "b",
    }
    ids, prompt_hash = _encode(Tokenizer(), row, row["options"][0], 4096)
    prompt = bytes(ids).decode()
    assert all(value in prompt for value in ("STATE", "QUESTION", "ANSWER"))
    assert "label" not in prompt and row["options"][1]["description"] not in prompt
    assert len(prompt_hash) == 64


def test_no_silent_truncation():
    row = {"id": "case", "state": "x" * 100, "question": "q", "options": [{"id": "a", "description": "a"}]}
    with pytest.raises(ValueError, match="no truncation|exceed"):
        _encode(Tokenizer(), row, row["options"][0], 10)


def test_official_yes_no_token_contract_is_checked():
    assert _answer_ids(AnswerTokenizer()) == (4, 7)


def test_reranker_rejects_image_inputs(tmp_path):
    from semif_phase1.reranker import score

    row = {
        "id": "x",
        "state": "owned evidence",
        "question": "Which answer follows?",
        "options": [
            {"id": "yes", "description": "Yes."},
            {"id": "no", "description": "No."},
        ],
        "Image": str(tmp_path / "shot.png"),
    }
    (tmp_path / "shot.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
        b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    with pytest.raises(ValueError, match="Image"):
        score(None, None, row, {})
