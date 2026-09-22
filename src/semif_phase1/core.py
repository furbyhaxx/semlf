"""Shared input validation, prompts, model loading, and numeric helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def validate_row(row: dict) -> None:
    required = {"id", "state", "question", "options"}
    if not required <= row.keys():
        raise ValueError(f"Row is missing fields: {sorted(required - row.keys())}")
    if not all(isinstance(row[key], str) and row[key] for key in ("id", "question")):
        raise ValueError("id and question must be nonempty strings")
    state = row["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("state must be a nonempty string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("state must be finite JSON-compatible data") from error
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS):
        raise ValueError("options must contain 2-16 entries")
    ids = []
    for option in options:
        if not isinstance(option, dict) or not isinstance(option.get("id"), str) or not isinstance(option.get("description"), str):
            raise ValueError("Each option needs string id and description fields")
        ids.append(option["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Option IDs must be unique")
    if "Image" in row:
        image = row["Image"]
        if not isinstance(image, str) or not image:
            raise ValueError("Image must be a nonempty path string")
        if not Path(image).is_file():
            raise ValueError(f"Image path does not exist or is not a file: {image}")


def user_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("User content must be a string or multimodal list")
    texts = [part.get("text") for part in content if isinstance(part, dict) and part.get("type") == "text"]
    if len(texts) != 1 or not isinstance(texts[0], str) or not texts[0]:
        raise ValueError("Multimodal user content needs exactly one text part")
    return texts[0]


def load_images(row: dict):
    if "Image" not in row:
        return None
    from PIL import Image as PILImage

    image = PILImage.open(row["Image"]).convert("RGB")
    image.load()
    return [image]


def tokenize_rendered(tokenizer, text: str, row: dict) -> tuple[list[int], dict]:
    images = load_images(row)
    processor = getattr(tokenizer, "processor", None)
    if not images:
        return tokenizer.encode(text, add_special_tokens=False), {}
    if processor is None:
        raise ValueError("Image inputs require a multimodal processor")
    batch = processor(text=[text], images=images, return_tensors="pt")
    token_ids = batch["input_ids"][0]
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    vision = {key: value for key, value in batch.items() if key != "input_ids"}
    return ids, vision


def prompt_digest(text: str, row: dict) -> str:
    if "Image" not in row:
        return digest(text)
    hasher = hashlib.sha256(text.encode())
    hasher.update(b"\0")
    hasher.update(Path(row["Image"]).read_bytes())
    return hasher.hexdigest()


def direct_messages(row: dict) -> list[dict]:
    validate_row(row)
    payload = json.dumps(
        {
            "evidence": row["state"],
            "criterion": row["question"],
            "options": [
                {"letter": LETTERS[index], "description": option["description"]}
                for index, option in enumerate(row["options"])
            ],
        },
        ensure_ascii=False,
    )
    if "Image" in row:
        user = [{"type": "image", "image": row["Image"]}, {"type": "text", "text": payload}]
    else:
        user = payload
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": user},
    ]


def softmax(values: list[float]) -> list[float]:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("Need at least two finite scores")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_causal_model(source: str, revision: str):
    """Load one pinned causal model on the sole visible CUDA device."""
    import torch
    import transformers

    local = Path(source).exists()
    if not local and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Remote models require a pinned 40-character commit revision")
    if local and not revision:
        raise ValueError("Local models require an explicit manifest/revision string")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one CUDA GPU, for example with CUDA_VISIBLE_DEVICES")
    common = {"revision": None if local else revision, "local_files_only": local, "trust_remote_code": False}
    config = transformers.AutoConfig.from_pretrained(source, **common)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    cls = transformers.AutoModelForCausalLM
    if config.model_type == "qwen3_5" and getattr(config, "vision_config", None) is not None:
        cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
        if cls is None:
            raise RuntimeError("Installed transformers lacks the native Qwen3.5 multimodal model")
        processor = transformers.AutoProcessor.from_pretrained(source, **common)
        tokenizer = getattr(processor, "tokenizer", tokenizer)
        tokenizer.processor = processor
    elif config.model_type in {"qwen3_5", "qwen3_5_text"}:
        cls = getattr(transformers, "Qwen3_5ForCausalLM", None)
        if cls is None:
            raise RuntimeError("Installed transformers lacks the native Qwen3.5 model")
        config = config.get_text_config()
    model, loading = cls.from_pretrained(
        source,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        output_loading_info=True,
        **common,
    )
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Checkpoint did not load completely: {loading}")
    model.eval()
    metadata = {
        "source": source,
        "revision": revision,
        "dtype": "bfloat16",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    return model, tokenizer, metadata
