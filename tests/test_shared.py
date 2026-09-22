import pytest

from semif_phase1.shared import _suffix_layout


def test_suffix_padding_follows_real_tokens():
    layout, ends = _suffix_layout([[3, 4], [5]], 7, 0)
    assert layout["input_ids"] == [[3, 4], [5, 0]]
    assert layout["attention_mask"] == [[1] * 9, [1] * 8 + [0]]
    assert layout["position_ids"] == [[7, 8], [7, 0]]
    assert ends == [1, 0]


def test_empty_suffix_is_rejected():
    with pytest.raises(ValueError):
        _suffix_layout([[1], []], 7, 0)


def test_shared_rejects_mixed_images(tmp_path):
    from semif_phase1.shared import score_shared

    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
        b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(png)
    second.write_bytes(png)
    options = [{"id": "yes", "description": "Yes."}, {"id": "no", "description": "No."}]
    rows = [
        {"id": "one", "state": "owned evidence", "question": "Q?", "options": options, "Image": str(first)},
        {"id": "two", "state": "owned evidence", "question": "Q?", "options": options, "Image": str(second)},
    ]
    with pytest.raises(ValueError, match="exact state"):
        score_shared(None, None, rows, {})
