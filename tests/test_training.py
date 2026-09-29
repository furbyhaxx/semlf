import json
import random

import pytest

from semif_phase1.core import direct_messages
from semif_phase1.training import (
    IGNORE_INDEX,
    answer_slot_features,
    augment,
    check_disjoint,
    permute_options,
    wanli_training_rows,
)

from test_direct import Tokenizer

OPTIONS = [
    {"id": "supported", "description": "The evidence establishes the claim"},
    {"id": "insufficient", "description": "The evidence does not establish either"},
    {"id": "contradicted", "description": "The evidence establishes the opposite"},
]
ROW = {"id": "r1", "group_id": "g1", "state": "owned evidence", "question": "Claim?", "options": OPTIONS, "label": 2}


def test_features_supervise_only_the_gold_letter():
    tokenizer = Tokenizer()
    features = answer_slot_features(tokenizer, ROW)
    prompt = tokenizer.apply_chat_template(direct_messages(ROW), tokenize=False)
    assert features["input_ids"] == tokenizer.encode(prompt) + [ord("C")]
    assert features["labels"][-1] == ord("C")
    assert set(features["labels"][:-1]) == {IGNORE_INDEX}
    assert len(features["attention_mask"]) == len(features["input_ids"])


def test_features_reject_images_and_bad_labels():
    with pytest.raises(ValueError, match="text rows"):
        answer_slot_features(Tokenizer(), dict(ROW, Image="x.png"))
    with pytest.raises(ValueError, match="label"):
        answer_slot_features(Tokenizer(), dict(ROW, label=3))


def test_permutation_keeps_the_gold_option():
    rng = random.Random(0)
    for _ in range(20):
        row = permute_options(ROW, rng)
        assert row["options"][row["label"]]["id"] == "contradicted"
        assert sorted(o["id"] for o in row["options"]) == sorted(o["id"] for o in OPTIONS)


def test_augment_ids_are_unique_and_original_first():
    rows = augment([ROW], 3)
    assert [row["id"] for row in rows] == ["r1", "r1~perm1", "r1~perm2"]
    assert rows[0] is ROW


def test_disjoint_check_catches_groups_and_repeated_text():
    check_disjoint([ROW], [dict(ROW, id="other", group_id="g2", state="different")])
    with pytest.raises(ValueError, match="group"):
        check_disjoint(augment([ROW], 2), [dict(ROW, id="x", group_id="g1/stability", state="other")])
    with pytest.raises(ValueError, match="repeats"):
        check_disjoint([ROW], [dict(ROW, id="x", group_id="g9")])


def test_wanli_rows_are_balanced_and_skip_evaluation_seeds(tmp_path):
    source = tmp_path / "train.jsonl"
    selection = tmp_path / "selection.jsonl"
    gold = ["entailment", "neutral", "contradiction"]
    upstream = [
        {"id": index, "pairID": str(1000 + index), "premise": f"premise {index}",
         "hypothesis": f"hypothesis {index}", "gold": gold[index % 3]}
        for index in range(30)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in upstream))
    selection.write_text(json.dumps({"source": "wanli", "upstream": {"seed_id": "1000"}}) + "\n")
    rows = wanli_training_rows(source, selection, per_label=4)
    assert len(rows) == 12
    assert "wanli-train-0" not in {row["id"] for row in rows}
    counts = {}
    for row in rows:
        counts[row["options"][row["label"]]["id"]] = counts.get(row["options"][row["label"]]["id"], 0) + 1
        assert row["question"].startswith("Assess the claim using only the supplied evidence: ")
    assert counts == {"supported": 4, "insufficient": 4, "contradicted": 4}
    assert rows == wanli_training_rows(source, selection, per_label=4)
