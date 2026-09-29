# Colab notebooks

Three notebooks run SemIf on Google Colab (a free T4 is enough for Qwen3.5-4B) or any
single CUDA GPU with compute capability 7.0 or newer. They load models through
[Unsloth](https://github.com/unslothai/unsloth) and keep the frozen `direct-options-v1`
prompt and answer-slot readout, so results stay comparable with the committed BF16
predictions. Nothing shells out to vLLM or another server: every decision is one forward
pass whose option logits are read directly.

| Notebook | Purpose | |
| --- | --- | --- |
| [SemIf_Inference](SemIf_Inference.ipynb) | Score single decisions, JSONL files, shared-state criteria and image rows | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/furbyhaxx/semlf/blob/master/notebooks/SemIf_Inference.ipynb) |
| [SemIf_Eval](SemIf_Eval.ipynb) | Quality, speed and memory per model and precision; drift from the reference; sweeps | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/furbyhaxx/semlf/blob/master/notebooks/SemIf_Eval.ipynb) |
| [SemIf_Train](SemIf_Train.ipynb) | LoRA / QLoRA on the answer-letter logits, evaluated before and after | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/furbyhaxx/semlf/blob/master/notebooks/SemIf_Train.ipynb) |

## Where the pieces come from

- **Install cell:** copied from Unsloth's `nb/Qwen3_5_(4B)_Vision.ipynb` in
  [unslothai/notebooks](https://github.com/unslothai/notebooks). That upstream notebook
  covers the same Qwen3.5-4B checkpoint and is tested on a T4. It installs Unsloth's
  bundled gated-delta-net kernels and `causal_conv1d` for the linear-attention layers.
- **SemIf** goes on `sys.path` and is not pip-installed. The `pyproject.toml` pins
  torch 2.10, which would replace the torch Unsloth was installed against.
- **`src/semif_phase1/unsloth_backend.py`** loads models through Unsloth. Hub models are
  resolved to an exact commit, and `use_exact_model_name` stops a 4-bit load from
  switching to another repository. The module also does left-padded batching, which
  halves the batch on out-of-memory errors, and it compares runs against a reference.
- **`src/semif_phase1/training.py`** builds training features from the exact scoring
  prompt with only the gold answer letter supervised. It also builds the class-balanced
  WANLI *train* rows, skipping every seed behind the frozen WANLI test rows, and
  permutes option order.
- **`notebooks/colab_helpers.py`** holds code shared by the eval and train notebooks:
  suites, SHA-256-checked downloads, metrics from `benchmarks/evaluate.py`, and
  create-only outputs.

## Keeping accuracy on older or smaller cards

1. Run **SemIf_Eval** at `16bit` first. On a T4 this runs in fp16, because bf16 needs
   Ampere or newer. `argmax_agree` against the committed BF16 predictions shows what
   the card alone changes.
2. Try `bnb-4bit` or `bnb-8bit` and larger `BATCH_SIZE` values in the sweep cell.
   `pick_config` returns the fastest setting that stays within `MAX_DROP` balanced
   accuracy of the best run. `delta_95ci` gives the paired bootstrap interval against
   the reference.
3. If 4-bit loses too much, train a QLoRA adapter in **SemIf_Train** at the same
   precision. Keep it only if `authored144` and `perturbations108` recover.

Base models other than Qwen3.5-4B work through the same cells. Pick one from the list or
type any Hub ID. The chat template must render each answer letter as exactly one token;
the scorer refuses models where it does not. `slot_mass` is the share of probability
that lands on the answer letters, and a low value means the model is not following the
answer format.

Batching was checked on CPU with tiny random-weight Qwen3 and Qwen3.5 models, using the
real transformers code. Left-padded batches matched single-row scoring to within 1e-6.
Unsloth's patched GPU kernels can differ, so the eval notebook repeats this check on the
real model before reporting any batched numbers.

## Editing

The `.ipynb` files are generated. Edit `build_notebooks.py`, then run

```bash
python notebooks/build_notebooks.py
pytest -q tests/test_notebooks.py
```

`tests/test_notebooks.py` fails if a committed notebook is stale or a code cell does not
compile.
