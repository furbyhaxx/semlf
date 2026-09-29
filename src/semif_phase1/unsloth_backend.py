"""Unsloth loading and batched option-logit readout for small or older CUDA GPUs.

Scoring keeps the frozen `direct-options-v1` prompt and answer slots, so the
results stay comparable with the committed BF16 reference predictions. Nothing
is generated: every decision is one forward pass read at the last position.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import re
import time
from pathlib import Path

from .core import softmax, validate_row

BATCH_SERVING_CONFIG = "unsloth-left-padded-batch-v1"
READOUT = "native full-vocabulary last-position logits restricted to declared answer slots"
PROBABILITY_STATUS = "conditional option score; uncalibrated as decision confidence"


def _unwrap_tokenizer(processor):
    """Return the text tokenizer, keeping a multimodal processor attached for Image rows."""
    inner = getattr(processor, "tokenizer", None)
    if inner is not None and type(processor).__name__.endswith("Processor"):
        inner.processor = processor
        return inner
    return processor


def _resolve_revision(source: str, revision: str | None, token=None) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        return revision
    from huggingface_hub import model_info

    return model_info(source, revision=revision or None, token=token).sha


def _supported(function, names: dict) -> dict:
    """Keep keywords this Unsloth release accepts; open **kwargs reach transformers."""
    parameters = inspect.signature(function).parameters
    open_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    return {key: value for key, value in names.items() if key in parameters or (open_kwargs and key != "use_exact_model_name")}


def weight_mode(load_in_4bit: bool, load_in_8bit: bool) -> str:
    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose at most one of load_in_4bit and load_in_8bit")
    return "bnb-4bit" if load_in_4bit else "bnb-8bit" if load_in_8bit else "16bit"


def suggest_weights(parameter_count: int, total_bytes: int, headroom: float = 0.8) -> str:
    """Pick the widest weight format whose weights plus ~1.5 GiB of activations fit."""
    budget = total_bytes * headroom - 1.5 * 2**30
    if 2 * parameter_count <= budget:
        return "16bit"
    if 0.6 * parameter_count <= budget:
        return "bnb-4bit"
    raise ValueError("The model does not fit this GPU even in 4-bit weights")


def gpu_profile() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; in Colab use Runtime > Change runtime type")
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_bytes": props.total_memory,
        "total_gib": round(props.total_memory / 2**30, 2),
        "bf16": torch.cuda.is_bf16_supported(),
    }


def load_model(
    source: str,
    revision: str | None = None,
    *,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    max_seq_length: int = 4096,
    dtype=None,
    for_training: bool = False,
    token=None,
    **extra,
):
    """Load a base checkpoint or a saved LoRA adapter directory through Unsloth.

    Remote checkpoints are resolved to an exact commit and loaded under their
    own name, so a 4-bit load does not silently switch to another repository.
    Returns `(model, tokenizer, metadata)` compatible with the torch scorers.
    """
    import unsloth  # noqa: F401  Unsloth must patch transformers before it loads.
    from unsloth import FastLanguageModel, FastVisionModel
    import torch
    import transformers

    weights = weight_mode(load_in_4bit, load_in_8bit)
    path = Path(source)
    adapter = path.is_dir() and (path / "adapter_config.json").is_file()
    if adapter:
        base = json.loads((path / "adapter_config.json").read_text())["base_model_name_or_path"]
        manifest = path / "semif_training.json"
        declared = json.loads(manifest.read_text()).get("base_revision") if manifest.is_file() else None
        config_source, resolved = base, None
    elif path.exists():
        if not revision:
            raise ValueError("Local checkpoints need an explicit revision label")
        config_source, resolved = source, revision
    else:
        config_source = source
        resolved = _resolve_revision(source, revision, token)
    config_revision = None if adapter or path.exists() else resolved
    config = transformers.AutoConfig.from_pretrained(
        config_source, revision=config_revision, token=token, trust_remote_code=False
    )
    vision = getattr(config, "vision_config", None) is not None
    loader = FastVisionModel if vision else FastLanguageModel
    requested = {
        "model_name": source,
        "max_seq_length": max_seq_length,
        "dtype": dtype,
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
        "token": token,
        "use_exact_model_name": True,
        "trust_remote_code": False,
        **({"revision": resolved} if resolved and not path.exists() else {}),
        **extra,
    }
    model, processor = loader.from_pretrained(**_supported(loader.from_pretrained, requested))
    tokenizer = _unwrap_tokenizer(processor)
    if not for_training:
        loader.for_inference(model)
    props = torch.cuda.get_device_properties(0)
    metadata = {
        "source": base if adapter else source,
        "revision": resolved or "unpinned-adapter-base",
        "adapter": str(path) if adapter else None,
        **({"adapter_base_revision_declared": declared} if adapter else {}),
        "loaded_name": getattr(model.config, "_name_or_path", source),
        "backend": "unsloth",
        "loader": loader.__name__,
        "weights": weights,
        "dtype": str(getattr(model, "dtype", dtype)).replace("torch.", ""),
        "max_seq_length": max_seq_length,
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "unsloth_version": getattr(unsloth, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    return model, tokenizer, metadata


def for_inference(model) -> None:
    """Switch an Unsloth model (plain or PEFT-wrapped) to inference mode."""
    from unsloth import FastLanguageModel, FastVisionModel

    vision = getattr(model.config, "vision_config", None) is not None
    (FastVisionModel if vision else FastLanguageModel).for_inference(model)


@contextlib.contextmanager
def _return_logits():
    """Unsloth may drop logits to save memory; readout needs them."""
    previous = os.environ.get("UNSLOTH_RETURN_LOGITS")
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("UNSLOTH_RETURN_LOGITS", None)
        else:
            os.environ["UNSLOTH_RETURN_LOGITS"] = previous


def _last_logit_kwargs(model) -> dict:
    target = model.get_base_model() if hasattr(model, "get_base_model") else model
    parameters = inspect.signature(target.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in parameters:
            return {name: 1}
    return {}


def left_pad(sequences: list[list[int]], pad_id: int) -> tuple[list[list[int]], list[list[int]]]:
    """Left-pad so every sequence's answer position is the final column."""
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Every decision needs a nonempty prompt")
    width = max(map(len, sequences))
    ids = [[pad_id] * (width - len(s)) + s for s in sequences]
    masks = [[0] * (width - len(s)) + [1] * len(s) for s in sequences]
    return ids, masks


def _result(row, ids, slots, prompt_hash, vocabulary, metadata, extra):
    import torch

    selected = vocabulary[slots].tolist()
    mass = torch.exp(torch.logsumexp(vocabulary[slots], 0) - torch.logsumexp(vocabulary, 0)).item()
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(selected),
        "option_logits": selected,
        "allowed_token_mass": mass,
        "input_tokens": len(ids),
        "prompt_sha256": prompt_hash,
        "prompt_version": "direct-options-v1",
        "model": metadata,
        "readout": READOUT,
        "probability_status": PROBABILITY_STATUS,
        **extra,
    }


def score_batched(model, tokenizer, rows: list[dict], metadata: dict, batch_size: int = 8,
                  max_tokens: int = 4096, progress=None) -> list[dict]:
    """Score rows in length-sorted, left-padded batches; halve the batch on OOM.

    Image rows go through the single-row multimodal scorer. Results keep input
    order; `forward_seconds` is the batch forward time divided by its rows.
    """
    import torch

    from .direct import encode_decision, score as direct_score

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Decision IDs must be unique")
    results: list[dict | None] = [None] * len(rows)
    pending = []
    with _return_logits():
        for index, row in enumerate(rows):
            validate_row(row)
            if "Image" in row:
                results[index] = {**direct_score(model, tokenizer, row, metadata, max_tokens),
                                  "serving_config": "direct-single-row-image"}
            else:
                ids, slots, prompt_hash, _ = encode_decision(tokenizer, row, max_tokens)
                pending.append((index, ids, slots, prompt_hash))
        pending.sort(key=lambda item: -len(item[1]))
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad is None:
            raise ValueError("Tokenizer requires a padding or EOS token")
        device = next(model.parameters()).device
        sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
        keep = _last_logit_kwargs(model)
        start, size = 0, batch_size
        while start < len(pending):
            chunk = pending[start:start + size]
            ids, masks = left_pad([item[1] for item in chunk], pad)
            try:
                sync()
                mark = time.perf_counter()
                with torch.inference_mode():
                    output = model(
                        input_ids=torch.tensor(ids, dtype=torch.long, device=device),
                        attention_mask=torch.tensor(masks, dtype=torch.long, device=device),
                        use_cache=False,
                        return_dict=True,
                        **keep,
                    )
                    vocabulary = output.logits[:, -1, :].float()
                    del output
                sync()
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
                torch.cuda.empty_cache()
                size = max(1, size // 2)
                continue
            seconds = time.perf_counter() - mark
            for row_index, (index, row_ids, slots, prompt_hash) in enumerate(chunk):
                results[index] = _result(
                    rows[index], row_ids, slots, prompt_hash, vocabulary[row_index].cpu(), metadata,
                    {
                        "forward_seconds": seconds / len(chunk),
                        "batch_forward_seconds": seconds,
                        "batch_rows": len(chunk),
                        "batch_padded_tokens": len(ids[0]) * len(chunk),
                        "serving_config": BATCH_SERVING_CONFIG,
                    },
                )
            del vocabulary
            start += len(chunk)
            if progress:
                progress(len(chunk))
    return results


def score_rows(model, tokenizer, rows: list[dict], metadata: dict, mode: str = "batched",
               batch_size: int = 8, max_tokens: int = 4096, progress=None) -> list[dict]:
    """Dispatch to batched, direct, serial, or shared scoring; results keep input order."""
    from .direct import score as direct_score
    from .serial import SerialPrefixScorer
    from .shared import score_shared

    if mode == "batched":
        return score_batched(model, tokenizer, rows, metadata, batch_size, max_tokens, progress)
    results = []
    with _return_logits():
        if mode == "direct":
            for row in rows:
                results.append(direct_score(model, tokenizer, row, metadata, max_tokens))
                if progress:
                    progress(1)
        elif mode == "serial":
            scorer = SerialPrefixScorer(model, tokenizer, metadata, max_tokens)
            for row in rows:
                results.append(scorer.score(row))
                if progress:
                    progress(1)
        elif mode == "shared":
            groups: dict[str, list[int]] = {}
            for index, row in enumerate(rows):
                key = json.dumps([row["state"], row.get("Image")], ensure_ascii=False, sort_keys=True)
                groups.setdefault(key, []).append(index)
            ordered: list[dict | None] = [None] * len(rows)
            for indices in groups.values():
                for chunk_start in range(0, len(indices), batch_size):
                    chunk = indices[chunk_start:chunk_start + batch_size]
                    scored, timing = score_shared(model, tokenizer, [rows[i] for i in chunk], metadata, max_tokens)
                    for index, result in zip(chunk, scored):
                        ordered[index] = {**result, "shared_timing": timing}
                    if progress:
                        progress(len(chunk))
            results = ordered
        else:
            raise ValueError(f"Unknown mode {mode!r}")
    return results


def _distribution(prediction: dict) -> dict:
    return dict(zip(prediction["option_ids"], prediction["probabilities"]))


def compare_predictions(reference: list[dict], candidate: list[dict]) -> dict:
    """Measure how far candidate option distributions drift from a reference run."""
    ref = {row["id"]: row for row in reference}
    cand = {row["id"]: row for row in candidate}
    shared = [key for key in ref if key in cand]
    if not shared:
        raise ValueError("No shared prediction IDs")
    agree, deltas, max_deltas, flips = 0, [], [], []
    for key in shared:
        left, right = _distribution(ref[key]), _distribution(cand[key])
        if set(left) != set(right):
            raise ValueError(f"Option IDs differ for {key}")
        left_top, right_top = max(left, key=left.get), max(right, key=right.get)
        agree += left_top == right_top
        if left_top != right_top:
            flips.append({"id": key, "reference": left_top, "candidate": right_top,
                          "reference_p": left[left_top], "candidate_p": right[right_top]})
        diffs = [abs(left[option] - right[option]) for option in left]
        deltas.append(sum(diffs) / 2)
        max_deltas.append(max(diffs))
    return {
        "n": len(shared),
        "missing_from_candidate": len(ref.keys() - cand.keys()),
        "argmax_agreement": agree / len(shared),
        "argmax_flips": len(flips),
        "mean_total_variation": sum(deltas) / len(deltas),
        "max_abs_probability_delta": max(max_deltas),
        "flips": sorted(flips, key=lambda item: item["id"]),
    }


def throughput(results: list[dict], wall_seconds: float) -> dict:
    """Summarize decisions per second and token counts for one scoring run."""
    if wall_seconds <= 0 or not results:
        raise ValueError("Need results and a positive wall time")
    tokens = sum(row["input_tokens"] for row in results)
    masses = [row["allowed_token_mass"] for row in results if "allowed_token_mass" in row]
    return {
        "decisions": len(results),
        "wall_seconds": wall_seconds,
        "decisions_per_second": len(results) / wall_seconds,
        "prompt_tokens_per_second": tokens / wall_seconds,
        "mean_input_tokens": tokens / len(results),
        "min_allowed_token_mass": min(masses) if masses else None,
    }
