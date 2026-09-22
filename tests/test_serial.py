from semif_phase1.core import user_text
from semif_phase1.serial import _state_prefix

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


class Processor:
    def __call__(self, text, images=None, return_tensors="pt", **kwargs):
        return {"input_ids": [list(text[0].encode())], "pixel_values": "pixels"}


class Tokenizer:
    def apply_chat_template(self, turns, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return "HEADER\n" + user_text(turns[-1]["content"]) + "\nASSISTANT"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode())


def test_state_prefix_stops_before_runtime_question_and_options():
    prefix = bytes(_state_prefix(Tokenizer(), "owned state")).decode()
    assert "owned state" in prefix
    assert "prefix boundary placeholder" not in prefix
    assert '"options"' not in prefix


def test_state_prefix_keeps_image_out_of_question_suffix(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(TINY_PNG)
    tokenizer = Tokenizer()
    tokenizer.processor = Processor()
    prefix = bytes(_state_prefix(tokenizer, "owned state", image=str(path))).decode()
    assert "owned state" in prefix
    assert "prefix boundary placeholder" not in prefix
    assert '"options"' not in prefix
