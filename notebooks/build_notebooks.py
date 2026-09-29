"""Generate the Colab notebooks from one source so their shared cells never drift.

    python notebooks/build_notebooks.py          # rewrite the .ipynb files
    python notebooks/build_notebooks.py --check  # fail if they are stale

The install cell follows unslothai/notebooks `nb/Qwen3_5_(4B)_Vision.ipynb`,
the upstream notebook for SemIf's reference model tested on a free T4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "furbyhaxx/semlf"


def colab(name: str) -> str:
    return f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name}"


def badge(name: str) -> str:
    url = colab(name)
    return f'<a href="{url}" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>'


INSTALL_MD = """### Installation

Same stack as Unsloth's Qwen3.5 (4B) notebook: Unsloth, its bundled gated-delta-net
kernels and `causal_conv1d` for the Qwen3.5 linear-attention layers. No vLLM:
every decision is one forward pass, so there is nothing to serve."""

INSTALL = """%%capture
import os, importlib.util
!pip install --upgrade -qqq uv
if importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):
    try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"
    except: _numpy = "numpy"; _pil = "pillow"
    !uv pip install -qqq \\
        "torch==2.8.0" "triton>=3.3.0" {_numpy} {_pil} torchvision bitsandbytes xformers==0.0.32.post2 \\
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\
        "unsloth[base] @ git+https://github.com/unslothai/unsloth"
    !uv pip install -qqq --no-deps "torchcodec==0.7.0"
elif importlib.util.find_spec("unsloth") is None:
    !uv pip install -qqq unsloth
!uv pip install --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" trl==0.22.2 unsloth unsloth_zoo
!uv pip install transformers==5.2.0
# Unsloth bundles the gated delta net kernels; a leftover pip fla would shadow them
!uv pip uninstall -qqq flash-linear-attention fla-core
# causal_conv1d is supported only on torch==2.8.0. If you have newer torch versions, please wait 10 minutes!
!uv pip install --no-build-isolation causal_conv1d==1.6.0
!uv pip install --no-deps --upgrade "torchao>=0.16.0\""""

SETUP = """# @title Fetch SemIf { display-mode: "form" }
# SemIf is put on sys.path instead of pip-installed: its pyproject pins torch 2.10,
# which would replace the torch Unsloth was installed against.
import json, subprocess, sys, time
from pathlib import Path

SEMLF_REPO = "https://github.com/furbyhaxx/semlf"  # @param {type:"string"}
SEMLF_REF = "main"  # @param {type:"string"}

ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src" / "semif_phase1").is_dir()), None)
if ROOT is None:
    ROOT = (Path("/content") if Path("/content").is_dir() else Path.cwd()) / "semlf"
    if not ROOT.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", SEMLF_REF, SEMLF_REPO, str(ROOT)], check=True)
for path in (ROOT / "src", ROOT / "notebooks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
COMMIT = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print(f"SemIf at {ROOT} ({COMMIT[:12] or 'not a git checkout'}); outputs go to {OUT}")"""

MODELS = """MODELS = {  # pinned revisions; any other Hub model or local path also works
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",  # reference: committed BF16 predictions
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3-4B-Instruct-2507": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "openbmb/MiniCPM5-2B": "12a3808a956f869c767195e9266b59c4d21d92e2",
}"""

MODEL_PARAMS = """MODEL = "Qwen/Qwen3.5-4B"  # @param ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B", "openbmb/MiniCPM5-2B"] {allow-input: true}
REVISION = ""  # @param {type:"string"}
WEIGHTS = "auto"  # @param ["auto", "16bit", "bnb-8bit", "bnb-4bit"]
MAX_SEQ_LENGTH = 4096  # @param {type:"integer"}"""

GPU_MD = """### GPU and precision

`WEIGHTS = "auto"` keeps 16-bit weights whenever they fit, because that preserves
the reference accuracy, and drops to 4-bit NF4 only when they do not. Choose it
yourself to trade accuracy for memory:

| Card | VRAM | Qwen3.5-4B suggestion |
| --- | --- | --- |
| A100 / L4 / RTX 3090-4090 | 22-80 GB | `16bit` in bf16, the reference setting |
| T4 / V100 / RTX 2080 Ti | 11-16 GB | `16bit` in fp16 (no bf16 before Ampere) |
| RTX 2060-3070, 8 GB | 6-8 GB | `bnb-4bit`, or a smaller base model |

`bnb-8bit` is the most faithful quantized format but is usually *slower* than
16-bit; `bnb-4bit` is smaller and faster than 8-bit. Unsloth's Triton kernels
need compute capability 7.0 or newer (V100, T4, RTX 20xx and later)."""

GPU = """import unsloth  # import before transformers so Unsloth can patch it
import torch
from huggingface_hub import model_info
from semif_phase1 import unsloth_backend as ub

GPU = ub.gpu_profile()
print(f"{GPU['name']}: {GPU['total_gib']} GiB, compute {GPU['compute_capability']}, bf16={GPU['bf16']}")
if float(GPU["compute_capability"]) < 7.0:
    print("WARNING: below compute 7.0; Unsloth's Triton kernels are unsupported on this card.")

revision = REVISION or MODELS.get(MODEL)
WEIGHTS_USED = WEIGHTS
if WEIGHTS == "auto":
    if Path(MODEL).exists():
        WEIGHTS_USED = "16bit"
    else:
        info = model_info(MODEL, revision=revision)
        params = info.safetensors.total if info.safetensors else None
        WEIGHTS_USED = ub.suggest_weights(params, GPU["total_bytes"]) if params else "16bit"
        print(f"{MODEL}: {params / 1e9:.2f}B parameters" if params else f"{MODEL}: parameter count unknown")
print(f"weights: {WEIGHTS_USED}")"""

LOAD = """model, tokenizer, META = ub.load_model(
    MODEL,
    revision,
    load_in_4bit = WEIGHTS_USED == "bnb-4bit",
    load_in_8bit = WEIGHTS_USED == "bnb-8bit",
    max_seq_length = MAX_SEQ_LENGTH,
)
META["semlf_commit"] = COMMIT
print(json.dumps(META, indent = 1))
print(f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated after loading")"""

FOOTER = """### Notes

* Probabilities are conditional option scores over the supplied options, not
  calibrated confidences. Validate them on your own workload before acting on them.
* Outputs are create-only, matching the repository's benchmark policy: every run
  writes a new file under `outputs/`.
* The Unsloth install cell follows
  [unslothai/notebooks](https://github.com/unslothai/notebooks) (LGPL-3.0)."""


def header(title: str, name: str, body: str) -> str:
    return f"""# SemIf · {title}

{badge(name)}

{body}

To run this, press *Runtime* → *Run all* on a **free** Tesla T4 Colab instance.
Companion notebooks: [inference]({colab(INFERENCE)}) · [evaluation]({colab(EVAL)}) · [training]({colab(TRAIN)})."""


INFERENCE, EVAL, TRAIN = "SemIf_Inference.ipynb", "SemIf_Eval.ipynb", "SemIf_Train.ipynb"


def common(title: str, name: str, body: str, config_extra: str = "") -> list[tuple[str, str]]:
    config = "# @title Model { display-mode: \"form\" }\n" + MODELS + "\n\n" + MODEL_PARAMS
    if config_extra:
        config += "\n" + config_extra
    return [
        ("markdown", header(title, name, body)),
        ("markdown", INSTALL_MD),
        ("code", INSTALL),
        ("code", SETUP),
        ("markdown", "### Configuration\n\nPick a pinned model from the list or type any Hub ID, "
                     "local checkpoint, or a LoRA adapter directory saved by the training notebook."),
        ("code", config),
        ("markdown", GPU_MD),
        ("code", GPU),
        ("markdown", "### Load through Unsloth\n\nQwen3.5 checkpoints are vision-language models and load "
                     "with `FastVisionModel`; text-only models use `FastLanguageModel`. The resolved "
                     "commit, precision and GPU are recorded in every result row."),
        ("code", LOAD),
    ]


def inference_cells() -> list[tuple[str, str]]:
    cells = common(
        "Inference",
        INFERENCE,
        "Score runtime-defined decisions by reading the model's option logits directly: one "
        "forward pass per decision, no generated tokens, no answer parsing. This notebook uses "
        "Unsloth's fast model loading and left-padded batching instead of a serving engine.",
        'BATCH_SIZE = 16  # @param {type:"integer"}\nINPUT_JSONL = ""  # @param {type:"string"}',
    )
    cells += [
        ("markdown", "### Ask one decision\n\n`decide` builds the same row format as "
                     "`examples/decisions.jsonl` and returns option probabilities."),
        ("code", """from semif_phase1.unsloth_backend import score_rows, throughput

def decide(state, question, options, image = None):
    row = {"id": "adhoc", "state": state, "question": question,
           "options": [{"id": key, "description": text} for key, text in options.items()]}
    if image:
        row["Image"] = str(image)
    result = score_rows(model, tokenizer, [row], META, mode = "direct")[0]
    return {key: round(p, 4) for key, p in zip(result["option_ids"], result["probabilities"])}

decide(
    "Customer cannot access an account after a password reset.",
    "Which queue should handle this request?",
    {"access": "Account access support.", "billing": "Billing support."},
)"""),
        ("markdown", "### Score a JSONL file in batches\n\nRows are sorted by length and left-padded so "
                     "the answer position is always the last column. If a batch runs out of memory it "
                     "is halved and retried, so a large `BATCH_SIZE` is safe on small cards."),
        ("code", """import pandas as pd
from tqdm.auto import tqdm
from colab_helpers import read_jsonl, write_create_only

source = Path(INPUT_JSONL) if INPUT_JSONL else ROOT / "examples" / "decisions.jsonl"
rows = read_jsonl(source)
bar = tqdm(total = len(rows), desc = source.name)
start = time.perf_counter()
results = score_rows(model, tokenizer, rows, META, mode = "batched", batch_size = BATCH_SIZE, progress = bar.update)
bar.close()
print(throughput(results, time.perf_counter() - start))

pd.DataFrame([
    {"id": r["id"], "choice": r["option_ids"][max(range(len(r["probabilities"])), key = r["probabilities"].__getitem__)],
     "p": round(max(r["probabilities"]), 4), "slot_mass": round(r.get("allowed_token_mass", float("nan")), 4),
     "tokens": r["input_tokens"]}
    for r in results
])"""),
        ("code", """path = write_create_only(OUT / f"scores-{source.stem}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl", results)
print(f"wrote {path}")"""),
        ("markdown", "`slot_mass` is the share of the full-vocabulary probability that lands on the answer "
                     "letters. Values far below 1 mean the model wanted to say something else, which is a "
                     "warning sign when trying smaller base models."),
        ("markdown", "### Many criteria over one state\n\nThe first 21 rows of the owned `shape777` fixture "
                     "share one 8 KB state. Compare fresh per-row scoring, batching, and `shared` mode, "
                     "which prefills the state once and branches every criterion from its KV cache."),
        ("code", """from semif_phase1.unsloth_backend import compare_predictions

group = []
with open(ROOT / "benchmarks" / "data" / "shape777.jsonl") as stream:
    for line in stream:
        row = json.loads(line)
        if group and row["state"] != group[0]["state"]:
            break
        group.append(row)

timings, by_mode = [], {}
for mode in ("direct", "batched", "shared"):
    try:
        score_rows(model, tokenizer, group[:2], META, mode = mode, batch_size = BATCH_SIZE)  # warm-up
        torch.cuda.synchronize()
        start = time.perf_counter()
        by_mode[mode] = score_rows(model, tokenizer, group, META, mode = mode, batch_size = BATCH_SIZE)
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        agreement = compare_predictions(by_mode["direct"], by_mode[mode])["argmax_agreement"]
        timings.append({"mode": mode, "seconds": round(wall, 3), "decisions/s": round(len(group) / wall, 2),
                        "argmax agreement with direct": agreement})
    except Exception as error:  # shared mode needs a native prefix cache that some patched models lack
        timings.append({"mode": mode, "error": f"{type(error).__name__}: {error}"[:200]})
pd.DataFrame(timings)"""),
        ("markdown", "### Image decisions\n\nQwen3.5 checkpoints read an optional local `Image` path "
                     "through their native processor. Image rows are scored one at a time."),
        ("code", """if META["loader"] == "FastVisionModel":
    image_rows = [dict(row, Image = str(ROOT / row["Image"])) for row in read_jsonl(ROOT / "examples" / "decisions-image.jsonl")]
    for result in score_rows(model, tokenizer, image_rows, META, mode = "direct"):
        print(result["id"], dict(zip(result["option_ids"], (round(p, 4) for p in result["probabilities"]))))
else:
    print(f"{MODEL} is text-only; skipping image rows")"""),
        ("markdown", FOOTER),
    ]
    return cells


SUITE_PARAMS = """BATCH_SIZE = 16  # @param {type:"integer"}
AUTHORED144 = True  # @param {type:"boolean"}
PERTURBATIONS108 = True  # @param {type:"boolean"}
WANLI256 = False  # @param {type:"boolean"}
EXTRA_SUITE_JSONL = ""  # @param {type:"string"}
COMPARE_REFERENCE = True  # @param {type:"boolean"}"""

SUITES = """from tqdm.auto import tqdm
import pandas as pd
import colab_helpers as ch

names = [name for name, keep in (("authored144", AUTHORED144), ("perturbations108", PERTURBATIONS108),
                                 ("wanli256", WANLI256)) if keep]
if EXTRA_SUITE_JSONL:
    names.append(EXTRA_SUITE_JSONL)
SUITES = ch.load_suites(names, OUT, ROOT)
print({name: len(rows) for name, rows in SUITES.items()})"""


def eval_cells() -> list[tuple[str, str]]:
    cells = common(
        "Evaluation",
        EVAL,
        "Measure decision quality, speed and memory for any model and precision on the owned "
        "benchmark suites, and check how far the results drift from the committed BF16 "
        "Qwen3.5-4B reference. Use it to find the cheapest setting for an older or smaller GPU "
        "that keeps accuracy, or to screen other base models.",
        SUITE_PARAMS,
    )
    cells += [
        ("markdown", "### Suites\n\n* `authored144`: the owned 144-row decision set behind the README's "
                     "0.813 balanced accuracy.\n* `perturbations108`: output-blind rewrites of the same "
                     "items (option order, wording).\n* `wanli256`: the frozen WANLI test selection, "
                     "fetched from its pinned Hub revision and SHA-256 checked.\n\nAny extra labeled "
                     "JSONL (rows with `label`, `family`, `group_id`) can be added as a suite."),
        ("code", SUITES),
        ("markdown", "### Padding parity\n\nBatching left-pads prompts. Before trusting batched numbers on "
                     "a new model or precision, confirm they match single-row scoring."),
        ("code", """from semif_phase1.unsloth_backend import compare_predictions, score_rows

probe = next(iter(SUITES.values()))[:32]
single = score_rows(model, tokenizer, probe, META, mode = "direct")
batched = score_rows(model, tokenizer, probe, META, mode = "batched", batch_size = BATCH_SIZE)
parity = compare_predictions(single, batched)
print({key: value for key, value in parity.items() if key != "flips"})
if parity["argmax_agreement"] < 1:
    print("WARNING: batching changed decisions on this model; set BATCH_SIZE = 1 for exact parity.")"""),
        ("markdown", "### Evaluate\n\n`delta_95ci` is this run's balanced accuracy minus the reference's, "
                     "with a paired source-group bootstrap interval; an interval that spans 0 means no "
                     "detectable change. `argmax_agree` and `mean_tv` (total variation) measure how "
                     "much individual decisions moved."),
        ("code", """def scorer(model, tokenizer, meta, batch_size):
    return lambda rows, progress: score_rows(model, tokenizer, rows, meta, mode = "batched",
                                             batch_size = batch_size, progress = progress)

REPORT, PREDICTIONS = ch.evaluate_suites(scorer(model, tokenizer, META, BATCH_SIZE), SUITES,
                                         COMPARE_REFERENCE, ROOT, tqdm)
pd.DataFrame(ch.summary_rows(REPORT, model = MODEL, weights = WEIGHTS_USED, batch = BATCH_SIZE))"""),
        ("code", """RUN_TAG = f"{MODEL.replace('/', '_')}-{WEIGHTS_USED}-{time.strftime('%Y%m%d-%H%M%S')}"
ch.write_create_only(OUT / f"eval-{RUN_TAG}.json", {"model": META, "gpu": GPU, "batch_size": BATCH_SIZE, "report": REPORT})
for name, rows in PREDICTIONS.items():
    ch.write_create_only(OUT / f"eval-{RUN_TAG}.{name}.predictions.jsonl", rows)
print(f"wrote {OUT}/eval-{RUN_TAG}.*")"""),
        ("markdown", "Decisions that flipped relative to the reference, if any:"),
        ("code", """pd.DataFrame([dict(suite = name, **flip) for name, entry in REPORT.items() for flip in entry.get("flips", [])])"""),
        ("markdown", "### Sweep precisions, batch sizes and base models\n\nEntries are "
                     "`model | weights | batch`, separated by `;`. The model loaded above is released first, then every "
                     "configuration is loaded, scored and freed in turn. `pick_config` returns the fastest "
                     "configuration within `MAX_DROP` balanced accuracy of the best one; set "
                     "`AGREEMENT_FLOOR` to also require argmax agreement with the Qwen3.5-4B reference "
                     "(leave it at 0 when comparing different base models)."),
        ("code", """RUN_SWEEP = False  # @param {type:"boolean"}
SWEEP = "Qwen/Qwen3.5-4B | 16bit | 16; Qwen/Qwen3.5-4B | bnb-8bit | 16; Qwen/Qwen3.5-4B | bnb-4bit | 16; Qwen/Qwen3.5-2B | 16bit | 16"  # @param {type:"string"}
SELECT_SUITE = "authored144"  # @param {type:"string"}
MAX_DROP = 0.02  # @param {type:"number"}
AGREEMENT_FLOOR = 0.0  # @param {type:"number"}

SWEEP_ROWS = []
if RUN_SWEEP:
    del model, tokenizer
    ch.release_cuda()
    for config in ch.parse_sweep(SWEEP, MODELS):
        print(f"--- {config}")
        model, tokenizer, meta = ub.load_model(
            config["model"], config["revision"], max_seq_length = MAX_SEQ_LENGTH,
            load_in_4bit = config["weights"] == "bnb-4bit", load_in_8bit = config["weights"] == "bnb-8bit")
        report, _ = ch.evaluate_suites(scorer(model, tokenizer, meta, config["batch_size"]), SUITES,
                                       COMPARE_REFERENCE, ROOT, tqdm)
        SWEEP_ROWS += ch.summary_rows(report, model = config["model"], weights = config["weights"],
                                      batch = config["batch_size"])
        del model, tokenizer
        ch.release_cuda()
    ch.write_create_only(OUT / f"sweep-{time.strftime('%Y%m%d-%H%M%S')}.json", {"gpu": GPU, "rows": SWEEP_ROWS})
    from IPython.display import display
    display(pd.DataFrame(SWEEP_ROWS))
    print("pick:", ch.pick_config(SWEEP_ROWS, SELECT_SUITE, MAX_DROP, AGREEMENT_FLOOR or None))"""),
        ("markdown", FOOTER),
    ]
    return cells


TRAIN_PARAMS = """LORA_R = 16  # @param {type:"integer"}
LORA_ALPHA = 16  # @param {type:"integer"}
WANLI_PER_LABEL = 500  # @param {type:"integer"}
OPTION_COPIES = 2  # @param {type:"integer"}
EXTRA_TRAIN_JSONL = ""  # @param {type:"string"}
MAX_TRAIN_TOKENS = 1024  # @param {type:"integer"}
MAX_STEPS = 150  # @param {type:"integer"}
EPOCHS = 1  # @param {type:"number"}
LEARNING_RATE = 1e-4  # @param {type:"number"}
PER_DEVICE_BATCH = 4  # @param {type:"integer"}
GRAD_ACCUM = 4  # @param {type:"integer"}
BATCH_SIZE = 16  # @param {type:"integer"}
EVAL_WANLI256 = False  # @param {type:"boolean"}
ADAPTER_NAME = ""  # @param {type:"string"}"""


def train_cells() -> list[tuple[str, str]]:
    cells = common(
        "Training",
        TRAIN,
        "Fine-tune a LoRA adapter for decision-native readout. Each example is the exact "
        "`direct-options-v1` scoring prompt, and the loss covers only the gold answer letter, "
        "so training optimizes the same last-position logits that inference reads. Use it to "
        "recover accuracy lost to 4-bit weights, to adapt a smaller base model, or to fit your "
        "own decision data.",
        TRAIN_PARAMS,
    )
    # Training loads its own weights; replace the shared load cell.
    load_index = next(i for i, (_, text) in enumerate(cells) if text == LOAD)
    cells[load_index] = ("code", """model, tokenizer, META = ub.load_model(
    MODEL,
    revision,
    load_in_4bit = WEIGHTS_USED == "bnb-4bit",
    load_in_8bit = WEIGHTS_USED == "bnb-8bit",
    max_seq_length = MAX_SEQ_LENGTH,
    for_training = True,
    use_gradient_checkpointing = "unsloth",
)
META["semlf_commit"] = COMMIT
from unsloth import FastLanguageModel, FastVisionModel
LOADER = FastVisionModel if META["loader"] == "FastVisionModel" else FastLanguageModel
print(json.dumps(META, indent = 1))""")
    cells[load_index - 1] = ("markdown", "### Load through Unsloth\n\n`16bit` gives 16-bit LoRA, which fits "
                             "Qwen3.5-4B on a T4 (Unsloth's own Qwen3.5 notebook does the same); `bnb-4bit` "
                             "gives QLoRA for 8 GB cards. Evaluate the adapter in the precision you will "
                             "serve it in.")
    cells += [
        ("markdown", "### Data\n\nBy default the training set is a class-balanced sample of the WANLI "
                     "**train** split, fetched from the same pinned revision as the WANLI test rows and "
                     "SHA-256 checked. Any WANLI seed that produced a frozen test row is dropped. Each "
                     "row is repeated `OPTION_COPIES` times with shuffled option order, so the letters "
                     "carry no answer prior.\n\nAdd your own rows with `EXTRA_TRAIN_JSONL`: the "
                     "`examples/decisions.jsonl` format plus an integer `label` indexing the gold "
                     "option. Evaluation suites are held out and checked for shared IDs, groups and "
                     "text."),
        ("code", """import colab_helpers as ch
from semif_phase1 import training

wanli_path = ch.download_verified(training.WANLI_TRAIN_URL, training.WANLI_TRAIN_SHA256,
                                  OUT / "sources" / "wanli-train.jsonl")
train_rows = []
if WANLI_PER_LABEL > 0:
    train_rows += training.wanli_training_rows(
        wanli_path, ROOT / "benchmarks" / "manifests" / "source-selection.jsonl", WANLI_PER_LABEL)
if EXTRA_TRAIN_JSONL:
    train_rows += ch.read_jsonl(EXTRA_TRAIN_JSONL)
train_rows = training.augment(train_rows, OPTION_COPIES)

SUITES = ch.load_suites(["authored144", "perturbations108"] + (["wanli256"] if EVAL_WANLI256 else []), OUT, ROOT)
training.check_disjoint(train_rows, *SUITES.values())
print(f"{len(train_rows)} training rows; held out: { {name: len(rows) for name, rows in SUITES.items()} }")"""),
        ("code", """from datasets import Dataset

features, skipped = [], 0
for row in train_rows:
    try:
        features.append(training.answer_slot_features(tokenizer, row, MAX_TRAIN_TOKENS))
    except ValueError:
        skipped += 1
train_dataset = Dataset.from_list(features).shuffle(seed = 3407)
print(f"{len(train_dataset)} examples, {skipped} skipped for length")
example = train_dataset[0]
print(repr(tokenizer.decode(example["input_ids"][-40:])))
print("supervised token:", repr(tokenizer.decode([t for t in example["labels"] if t != training.IGNORE_INDEX])))"""),
        ("markdown", "### Baseline\n\nScore the held-out suites before training, in the same precision, "
                     "so the before/after comparison isolates the adapter."),
        ("code", """from tqdm.auto import tqdm
import pandas as pd
from semif_phase1.unsloth_backend import score_rows

def scorer(batch_size):
    return lambda rows, progress: score_rows(model, tokenizer, rows, META, mode = "batched",
                                             batch_size = batch_size, progress = progress)

LOADER.for_inference(model)
BEFORE, _ = ch.evaluate_suites(scorer(BATCH_SIZE), SUITES, True, ROOT, tqdm)
pd.DataFrame(ch.summary_rows(BEFORE, stage = "before"))"""),
        ("markdown", "### LoRA adapters\n\nOnly language layers are adapted; the vision tower of "
                     "Qwen3.5 stays frozen because training rows are text-only."),
        ("code", """if LOADER is FastVisionModel:
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers = False,
        finetune_language_layers = True,
        finetune_attention_modules = True,
        finetune_mlp_modules = True,
        r = LORA_R,
        lora_alpha = LORA_ALPHA,
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
    )
else:
    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_R,
        lora_alpha = LORA_ALPHA,
        lora_dropout = 0,
        bias = "none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
model.print_trainable_parameters()"""),
        ("markdown", "### Train\n\nA plain `transformers.Trainer` with pre-tokenized features: the labels "
                     "mask everything except the answer letter, and Unsloth's patched model supplies "
                     "the fast LoRA kernels, gradient checkpointing and memory-light cross-entropy. "
                     "`MAX_STEPS = 150` fits a free Colab session; set it to `-1` to train for `EPOCHS`."),
        ("code", """import math
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from unsloth import is_bf16_supported

steps = MAX_STEPS if MAX_STEPS > 0 else math.ceil(len(train_dataset) / (PER_DEVICE_BATCH * GRAD_ACCUM) * EPOCHS)
LOADER.for_training(model)
trainer = Trainer(
    model = model,
    train_dataset = train_dataset,
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding = True, pad_to_multiple_of = 8,
                                           label_pad_token_id = training.IGNORE_INDEX),
    args = TrainingArguments(
        output_dir = str(OUT / "checkpoints"),
        per_device_train_batch_size = PER_DEVICE_BATCH,
        gradient_accumulation_steps = GRAD_ACCUM,
        warmup_steps = max(1, steps // 20),
        max_steps = MAX_STEPS,
        num_train_epochs = EPOCHS,
        learning_rate = LEARNING_RATE,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        weight_decay = 0.01,
        logging_steps = 5,
        save_strategy = "no",
        seed = 3407,
        report_to = "none",
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        remove_unused_columns = False,
    ),
)"""),
        ("code", """# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")"""),
        ("code", "trainer_stats = trainer.train()"),
        ("code", """# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
print(f"{trainer_stats.metrics['train_runtime']:.0f} seconds used for training.")
print(f"Peak reserved memory = {used_memory} GB ({used_memory_for_lora} GB for training).")
history = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
print(f"loss first -> last: {history[0]:.4f} -> {history[-1]:.4f}" if len(history) > 1 else history)"""),
        ("markdown", "### After training\n\nThe same suites, precision and batch size as the baseline. "
                     "Keep the adapter only if `authored144` and `perturbations108` hold up: WANLI is "
                     "a different distribution, and gains there can cost accuracy on your decisions."),
        ("code", """LOADER.for_inference(model)
AFTER, AFTER_PREDICTIONS = ch.evaluate_suites(scorer(BATCH_SIZE), SUITES, True, ROOT, tqdm)
pd.DataFrame(ch.summary_rows(BEFORE, stage = "before") + ch.summary_rows(AFTER, stage = "after"))"""),
        ("markdown", "### Save the adapter\n\nThis saves only the LoRA weights plus `semif_training.json`, "
                     "a manifest with the base revision, data, hyperparameters and before/after "
                     "metrics. Point the inference or evaluation notebook's `MODEL` at the directory "
                     "to use it."),
        ("code", """name = ADAPTER_NAME or f"lora-{MODEL.replace('/', '_')}-{WEIGHTS_USED}-{time.strftime('%Y%m%d-%H%M%S')}"
ADAPTER_DIR = OUT / name
if ADAPTER_DIR.exists():
    raise FileExistsError(f"{ADAPTER_DIR} exists; choose a new ADAPTER_NAME")
model.save_pretrained(str(ADAPTER_DIR))
getattr(tokenizer, "processor", tokenizer).save_pretrained(str(ADAPTER_DIR))
ch.write_create_only(ADAPTER_DIR / "semif_training.json", {
    "base_model": MODEL,
    "base_revision": META["revision"],
    "weights": WEIGHTS_USED,
    "semlf_commit": COMMIT,
    "gpu": GPU,
    "objective": "cross-entropy on the gold answer letter of the direct-options-v1 prompt",
    "data": {"wanli_train_sha256": training.WANLI_TRAIN_SHA256, "wanli_per_label": WANLI_PER_LABEL,
             "extra_train_jsonl": EXTRA_TRAIN_JSONL or None, "option_copies": OPTION_COPIES,
             "examples": len(train_dataset), "skipped_for_length": skipped},
    "hyperparameters": {"lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "learning_rate": LEARNING_RATE,
                        "max_steps": MAX_STEPS, "epochs": EPOCHS, "per_device_batch": PER_DEVICE_BATCH,
                        "grad_accum": GRAD_ACCUM, "max_train_tokens": MAX_TRAIN_TOKENS},
    "train_runtime_seconds": trainer_stats.metrics["train_runtime"],
    "before": BEFORE,
    "after": AFTER,
})
print(f"saved {ADAPTER_DIR}")
# model.push_to_hub("your_name/semif_lora", token = "YOUR_HF_TOKEN")"""),
        ("markdown", "### Merged 16-bit and GGUF exports\n\nOptional. A merged 16-bit checkpoint loads "
                     "without PEFT. GGUF targets llama.cpp and the browser demo's wllama runtime; "
                     "Unsloth builds llama.cpp for the conversion, and quantized GGUF should be "
                     "re-evaluated before use."),
        ("code", """if False: model.save_pretrained_merged(str(OUT / f"{name}-merged"), tokenizer)
if False: model.save_pretrained_gguf(str(OUT / f"{name}-gguf"), tokenizer, quantization_method = "q4_k_m")"""),
        ("markdown", FOOTER),
    ]
    return cells


def notebook(cells: list[tuple[str, str]]) -> dict:
    def source(text: str) -> list[str]:
        lines = text.split("\n")
        return [line + "\n" for line in lines[:-1]] + [lines[-1]]

    return {
        "cells": [
            {"cell_type": kind, "metadata": {}, "source": source(text),
             **({"execution_count": None, "outputs": []} if kind == "code" else {})}
            for kind, text in cells
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


NOTEBOOKS = {INFERENCE: inference_cells, EVAL: eval_cells, TRAIN: train_cells}


def render(name: str) -> str:
    return json.dumps(notebook(NOTEBOOKS[name]()), indent=1, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if a committed notebook is stale")
    args = parser.parse_args()
    stale = []
    for name in NOTEBOOKS:
        path, text = HERE / name, render(name)
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(name)
        else:
            path.write_text(text)
            print(f"wrote {path}")
    if stale:
        sys.exit(f"stale notebooks, run python notebooks/build_notebooks.py: {', '.join(stale)}")


if __name__ == "__main__":
    main()
