import math

import pytest

from semif_phase1.core import direct_messages, softmax, validate_row


ROW = {
    "id": "x",
    "state": "owned evidence",
    "question": "Which answer follows?",
    "options": [
        {"id": "yes", "description": "Yes."},
        {"id": "no", "description": "No."},
    ],
}


def test_direct_prompt_excludes_extra_fields():
    row = dict(ROW, label="yes", provenance={"secret": "do not leak"})
    rendered = str(direct_messages(row))
    assert "owned evidence" in rendered
    assert "secret" not in rendered
    assert "label" not in rendered


def test_softmax_is_finite_and_normalized():
    values = softmax([1000.0, 999.0, -1000.0])
    assert all(math.isfinite(value) for value in values)
    assert sum(values) == pytest.approx(1.0)
    assert values[0] > values[1] > values[2]


def test_duplicate_options_rejected():
    row = dict(ROW, options=[ROW["options"][0], ROW["options"][0]])
    with pytest.raises(ValueError, match="unique"):
        validate_row(row)


def test_structured_json_state_is_supported():
    row = dict(ROW, state={"policy": "Never request passwords", "candidate": ["invoice id"]})
    validate_row(row)
    assert '"policy"' in direct_messages(row)[1]["content"]


def test_nonfinite_structured_state_is_rejected():
    with pytest.raises(ValueError, match="finite JSON-compatible"):
        validate_row(dict(ROW, state={"score": float("nan")}))


TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_image_field_is_optional_and_unused_rows_stay_text():
    validate_row(ROW)
    content = direct_messages(ROW)[1]["content"]
    assert isinstance(content, str)
    assert "Image" not in content


def test_image_must_be_an_existing_file(tmp_path):
    with pytest.raises(ValueError, match="Image"):
        validate_row(dict(ROW, Image=""))
    with pytest.raises(ValueError, match="Image"):
        validate_row(dict(ROW, Image=str(tmp_path / "missing.png")))
    path = tmp_path / "shot.png"
    path.write_bytes(TINY_PNG)
    row = dict(ROW, Image=str(path))
    validate_row(row)
    user = direct_messages(row)[1]["content"]
    assert user[0] == {"type": "image", "image": str(path)}
    assert user[1]["type"] == "text"
    assert '"evidence": "owned evidence"' in user[1]["text"]
    assert str(path) not in user[1]["text"]
    assert direct_messages(ROW)[1]["content"] == user[1]["text"]
