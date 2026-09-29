import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))

import colab_helpers as ch  # noqa: E402


def python_source(cell: dict) -> str:
    """Replace IPython magics and shell escapes with `pass`, keeping indentation."""
    lines, continued = [], False
    for line in "".join(cell["source"]).splitlines():
        stripped = line.lstrip()
        if continued:
            continued = line.endswith("\\")
            continue
        if stripped.startswith(("!", "%")):
            continued = line.endswith("\\")
            line = line[: len(line) - len(stripped)] + "pass"
        lines.append(line)
    return "\n".join(lines)


def test_committed_notebooks_match_the_generator():
    result = subprocess.run([sys.executable, str(NOTEBOOKS / "build_notebooks.py"), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["SemIf_Inference.ipynb", "SemIf_Eval.ipynb", "SemIf_Train.ipynb"])
def test_every_code_cell_compiles(name):
    notebook = json.loads((NOTEBOOKS / name).read_text())
    assert notebook["metadata"]["colab"]["gpuType"] == "T4"
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code
    for index, cell in enumerate(code):
        compile(python_source(cell), f"{name}[{index}]", "exec")
    text = "".join("".join(cell["source"]) for cell in code)
    assert "vllm" not in text.lower() or "No vLLM" in text
    assert re.search(r"import unsloth", text)


def test_helpers_reproduce_the_published_reference_numbers():
    gold = ch.read_jsonl(ROOT / ch.SUITE_FILES["authored144"])
    reference = ch.reference_predictions("authored144")
    metrics = ch.quality(gold, reference)
    assert round(metrics["balanced_accuracy"], 3) == 0.813
    assert metrics["coverage"] == 1.0
    paired = ch.paired_difference(gold, reference, reference)
    assert paired["difference"] == 0


def test_suites_load_and_unknown_names_fail(tmp_path):
    suites = ch.load_suites(["authored144", "perturbations108"], tmp_path)
    assert {name: len(rows) for name, rows in suites.items()} == {"authored144": 144, "perturbations108": 108}
    with pytest.raises(ValueError, match="Unknown suite"):
        ch.load_suites(["nope"], tmp_path)


def test_sweep_parsing_and_selection():
    configs = ch.parse_sweep("A | 16bit | 8; B | bnb-4bit | 16\n", {"A": "rev"})
    assert configs == [
        {"model": "A", "revision": "rev", "weights": "16bit", "batch_size": 8},
        {"model": "B", "revision": None, "weights": "bnb-4bit", "batch_size": 16},
    ]
    with pytest.raises(ValueError):
        ch.parse_sweep("A | 3bit | 8", {})
    rows = [
        {"suite": "s", "bal_acc": 0.81, "argmax_agree": 1.0, "dec_per_s": 5},
        {"suite": "s", "bal_acc": 0.80, "argmax_agree": 0.95, "dec_per_s": 9},
        {"suite": "s", "bal_acc": 0.70, "argmax_agree": 0.80, "dec_per_s": 20},
    ]
    assert ch.pick_config(rows, "s", 0.02)["dec_per_s"] == 9
    assert ch.pick_config(rows, "s", 0.02, agreement_floor=0.97)["dec_per_s"] == 5


def test_outputs_are_create_only(tmp_path):
    path = ch.write_create_only(tmp_path / "run.json", {"a": 1})
    with pytest.raises(FileExistsError):
        ch.write_create_only(path, {"a": 2})
