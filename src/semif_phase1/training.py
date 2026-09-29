"""Answer-slot training rows and features that reuse the exact scoring prompt."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .core import validate_row

IGNORE_INDEX = -100
WANLI_DESCRIPTIONS = {
    "supported": "The evidence establishes the claim",
    "insufficient": "The evidence does not establish either",
    "contradicted": "The evidence establishes the opposite",
}
WANLI_LABELS = {"entailment": "supported", "neutral": "insufficient", "contradiction": "contradicted"}
WANLI_TRAIN_URL = (
    "https://huggingface.co/datasets/alisawuffles/WANLI/resolve/"
    "61c95318fd71c55b6ba355d76253254615f387ec/train.jsonl"
)
WANLI_TRAIN_SHA256 = "85058cf017a911e89242dc29fa0a4ddaad3664cb923dc0a82145fdda14b694e5"


def gold_id(row: dict) -> str:
    """Return the gold option ID of a labeled row."""
    label = row.get("label")
    if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(row["options"]):
        raise ValueError(f"Row {row.get('id')}: label must index one option")
    return row["options"][label]["id"]


def permute_options(row: dict, rng: random.Random) -> dict:
    """Shuffle option order and remap the label, so letters carry no answer prior."""
    target = gold_id(row)
    options = list(row["options"])
    rng.shuffle(options)
    return {**row, "options": options, "label": [option["id"] for option in options].index(target)}


def augment(rows: list[dict], copies: int, seed: int = 3407) -> list[dict]:
    """Return each row plus `copies - 1` option permutations, with unique IDs."""
    if copies < 1:
        raise ValueError("copies must be positive")
    rng, result = random.Random(seed), []
    for row in rows:
        result.append(row)
        for index in range(1, copies):
            result.append({**permute_options(row, rng), "id": f"{row['id']}~perm{index}"})
    return result


def answer_slot_features(tokenizer, row: dict, max_tokens: int = 4096) -> dict:
    """Tokenize the exact scoring prompt and supervise only the gold answer letter.

    Causal LMs shift labels by one position, so the loss lands on the same
    last-position logits that `direct.score` reads at inference time.
    """
    from .direct import encode_decision

    if "Image" in row:
        raise ValueError("Answer-slot training features support text rows only")
    gold_id(row)
    ids, slots, _, _ = encode_decision(tokenizer, row, max_tokens - 1)
    answer = slots[row["label"]]
    return {
        "input_ids": ids + [answer],
        "attention_mask": [1] * (len(ids) + 1),
        "labels": [IGNORE_INDEX] * len(ids) + [answer],
    }


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def excluded_wanli_seeds(selection_path) -> set[str]:
    """Seed IDs behind the frozen WANLI evaluation rows; training must avoid them."""
    return {
        str(item["upstream"]["seed_id"])
        for item in read_jsonl(selection_path)
        if item.get("source") == "wanli"
    }


def wanli_training_rows(source_path, selection_path, per_label: int, seed: int = 3407) -> list[dict]:
    """Convert a class-balanced sample of WANLI train into labeled decision rows.

    Rows share the evaluation's question wording and option descriptions, but
    any upstream seed that also produced a frozen test row is dropped.
    """
    if per_label < 1:
        raise ValueError("per_label must be positive")
    excluded = excluded_wanli_seeds(selection_path)
    by_label = defaultdict(list)
    for upstream in read_jsonl(source_path):
        if str(upstream["pairID"]) in excluded or upstream["gold"] not in WANLI_LABELS:
            continue
        by_label[WANLI_LABELS[upstream["gold"]]].append(upstream)
    rng, rows = random.Random(seed), []
    for key in sorted(WANLI_DESCRIPTIONS):
        pool = sorted(by_label[key], key=lambda item: item["id"])
        if len(pool) < per_label:
            raise ValueError(f"Only {len(pool)} WANLI train rows for {key}")
        for upstream in rng.sample(pool, per_label):
            option_ids = list(WANLI_DESCRIPTIONS)
            rng.shuffle(option_ids)
            rows.append(
                {
                    "id": f"wanli-train-{upstream['id']}",
                    "group_id": f"wanli-seed-{upstream['pairID']}",
                    "family": "evidence_interpretation",
                    "split": "train",
                    "state": upstream["premise"],
                    "question": "Assess the claim using only the supplied evidence: " + upstream["hypothesis"],
                    "options": [{"id": item, "description": WANLI_DESCRIPTIONS[item]} for item in option_ids],
                    "label": option_ids.index(key),
                    "provenance": {
                        "source": "WANLI",
                        "source_id": upstream["id"],
                        "source_seed_id": upstream["pairID"],
                        "source_official_split": "train",
                        "original_label": upstream["gold"],
                        "rights": "CC-BY-4.0",
                    },
                }
            )
    rng.shuffle(rows)
    return rows


def check_disjoint(train: list[dict], *held_out: list[dict]) -> None:
    """Refuse training rows whose ID, group, or exact (state, question) is held out."""
    for row in train:
        validate_row(row)
        gold_id(row)
    train_ids = {row["id"].split("~perm")[0] for row in train}
    train_groups = {row.get("group_id") for row in train} - {None}
    train_pairs = {json.dumps([row["state"], row["question"]], ensure_ascii=False) for row in train}
    for rows in held_out:
        for row in rows:
            base_group = str(row.get("group_id", "")).split("/")[0]
            if row["id"] in train_ids or row.get("group_id") in train_groups or base_group in train_groups:
                raise ValueError(f"Held-out row {row['id']} shares an ID or group with training data")
            if json.dumps([row["state"], row["question"]], ensure_ascii=False) in train_pairs:
                raise ValueError(f"Held-out row {row['id']} repeats a training state and question")
