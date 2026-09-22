from semif_phase1.core import direct_messages
from semif_phase1.direct import encode_decision, encode_prompt

ROW = {
    "id": "x",
    "state": "owned evidence",
    "question": "Which answer follows?",
    "options": [
        {"id": "yes", "description": "Yes."},
        {"id": "no", "description": "No."},
    ],
}
TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


class Tokenizer:
    def apply_chat_template(self, turns, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        content = turns[-1]["content"]
        if isinstance(content, list):
            text = next(part["text"] for part in content if part["type"] == "text")
            return "HEADER\n<image>\n" + text + "\nASSISTANT"
        return "HEADER\n" + content + "\nASSISTANT"

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode()


class Processor:
    def __call__(self, text, images=None, return_tensors="pt", **kwargs):
        body = text[0]
        ids = ([255] if images else []) + list(body.encode())
        encoded = {"input_ids": [ids]}
        if images:
            encoded["pixel_values"] = "pixels"
            encoded["image_grid_thw"] = "grid"
        return encoded


def test_text_encode_prompt_unchanged():
    tokenizer = Tokenizer()
    ids, slots, digest = encode_prompt(tokenizer, ROW, 4096)
    prompt = tokenizer.apply_chat_template(
        direct_messages(ROW), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert ids == tokenizer.encode(prompt)
    assert slots == [ord("A"), ord("B")]
    assert len(digest) == 64


def test_image_decision_uses_processor_and_keeps_letter_slots(tmp_path):
    path = tmp_path / "shot.png"
    path.write_bytes(TINY_PNG)
    tokenizer = Tokenizer()
    tokenizer.processor = Processor()
    row = dict(ROW, Image=str(path))
    ids, slots, digest, vision = encode_decision(tokenizer, row, 4096)
    assert ids[0] == 255
    assert "pixel_values" in vision and "image_grid_thw" in vision
    assert len(slots) == 2
    assert len(digest) == 64