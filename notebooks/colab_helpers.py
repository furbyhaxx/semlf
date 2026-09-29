"""Shared helpers for the Colab notebooks: suites, verified downloads, and eval reports.

The notebooks put this directory and `src/` on `sys.path`; nothing here is part
of the installed `semif_phase1` package.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE_FILES = {
    "authored144": "benchmarks/data/authored144.jsonl",
    "perturbations108": "benchmarks/data/perturbations108.jsonl",
}
REFERENCE_FILES = {
    "authored144": "results/raw/predictions/direct-authored144.jsonl",
    "perturbations108": "results/raw/predictions/direct-perturbations108.jsonl",
    "wanli256": "results/raw/predictions/direct-wanli256.jsonl",
}
REFERENCE_MODEL = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")


def benchmark_module(name: str, root: Path = ROOT):
    """Import `benchmarks/<name>.py` without clashing with same-named PyPI packages."""
    spec = importlib.util.spec_from_file_location(f"semif_benchmarks_{name}", root / "benchmarks" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def download_verified(url: str, sha256: str, destination: Path) -> Path:
    """Download once, refusing any bytes whose SHA-256 differs from the pin."""
    destination = Path(destination)
    if not destination.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "semif-colab/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha256:
            raise ValueError(f"{url} changed: expected {sha256}, received {actual}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    elif hashlib.sha256(destination.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"{destination} does not match its pinned SHA-256; delete it and rerun")
    return destination


def build_wanli256(out: Path, root: Path = ROOT) -> Path:
    """Fetch the pinned WANLI test split and rebuild the frozen 256-row evaluation."""
    target = Path(out) / "wanli256.jsonl"
    if target.exists():
        return target
    url, sha256 = benchmark_module("fetch_sources", root).SOURCES["wanli-test.jsonl"]
    source = download_verified(url, sha256, Path(out) / "sources" / "wanli-test.jsonl")
    subprocess.run(
        [sys.executable, str(root / "benchmarks" / "build_wanli.py"), "--source", str(source),
         "--selection", str(root / "benchmarks" / "manifests" / "source-selection.jsonl"), "--output", str(target)],
        check=True,
    )
    return target


def load_suites(names, out: Path, root: Path = ROOT) -> dict[str, list[dict]]:
    suites = {}
    for name in names:
        if name == "wanli256":
            suites[name] = read_jsonl(build_wanli256(out, root))
        elif name in SUITE_FILES:
            suites[name] = read_jsonl(root / SUITE_FILES[name])
        else:
            path = Path(name)
            if not path.is_file():
                raise ValueError(f"Unknown suite {name!r}; use a built-in name or a labeled JSONL path")
            suites[path.stem] = read_jsonl(path)
    return suites


def reference_predictions(name: str, root: Path = ROOT):
    path = root / REFERENCE_FILES[name] if name in REFERENCE_FILES else None
    return read_jsonl(path) if path and path.is_file() else None


def quality(gold: list[dict], predictions: list[dict], root: Path = ROOT) -> dict:
    """The metrics behind the README tables, plus mean NLL over all rows."""
    evaluator = benchmark_module("evaluate", root)
    report = evaluator.evaluate(gold, predictions)
    aligned = evaluator.align(gold, predictions)
    nll = [row["nll"] for row in aligned if row["nll"] is not None]
    return {
        "balanced_accuracy": report["mean_family_balanced_accuracy"],
        "macro_f1": report["mean_family_macro_f1"],
        "accuracy": sum(row["correct"] for row in aligned) / len(aligned),
        "coverage": report["coverage"],
        "mean_nll": sum(nll) / len(nll) if len(nll) == len(aligned) else None,
    }


def paired_difference(gold, predictions, reference, root: Path = ROOT) -> dict:
    """Candidate minus reference balanced accuracy with a paired source-group bootstrap CI."""
    evaluator = benchmark_module("evaluate", root)
    return evaluator.paired_comparison(evaluator.align(gold, predictions), evaluator.align(gold, reference))


def evaluate_suites(score, suites: dict, compare_reference: bool = True, root: Path = ROOT,
                    progress_factory=None) -> tuple[dict, dict]:
    """Score every suite with `score(rows, progress)` and return (report, predictions)."""
    import torch

    from semif_phase1.unsloth_backend import compare_predictions, throughput

    report, predictions = {}, {}
    for name, gold in suites.items():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        bar = progress_factory(total=len(gold), desc=name) if progress_factory else None
        start = time.perf_counter()
        scored = score(gold, bar.update if bar else None)
        wall = time.perf_counter() - start
        if bar:
            bar.close()
        entry = {**quality(gold, scored, root), **throughput(scored, wall),
                 "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
        reference = reference_predictions(name, root) if compare_reference else None
        if reference is not None:
            parity = compare_predictions(reference, scored)
            entry["reference_balanced_accuracy"] = quality(gold, reference, root)["balanced_accuracy"]
            entry["argmax_agreement"] = parity["argmax_agreement"]
            entry["mean_total_variation"] = parity["mean_total_variation"]
            entry["paired_difference"] = paired_difference(gold, scored, reference, root)
            entry["flips"] = parity["flips"]
        report[name], predictions[name] = entry, scored
    return report, predictions


def summary_rows(report: dict, **labels) -> list[dict]:
    """Flatten an evaluate_suites report into one display row per suite."""
    rows = []
    for name, entry in report.items():
        paired = entry.get("paired_difference", {})
        interval = paired.get("paired_source_group_bootstrap_95")
        rows.append({
            **labels,
            "suite": name,
            "bal_acc": round(entry["balanced_accuracy"], 3),
            "ref_bal_acc": round(entry["reference_balanced_accuracy"], 3) if "reference_balanced_accuracy" in entry else None,
            "delta_95ci": f"[{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval else None,
            "argmax_agree": round(entry["argmax_agreement"], 3) if "argmax_agreement" in entry else None,
            "mean_tv": round(entry["mean_total_variation"], 4) if "mean_total_variation" in entry else None,
            "nll": round(entry["mean_nll"], 3) if entry.get("mean_nll") is not None else None,
            "dec_per_s": round(entry["decisions_per_second"], 2),
            "peak_gib": round(entry["peak_gib"], 2),
            "min_slot_mass": round(entry["min_allowed_token_mass"], 3) if entry.get("min_allowed_token_mass") is not None else None,
        })
    return rows


def parse_sweep(text: str, default_revisions: dict) -> list[dict]:
    """Parse `model | weights | batch` entries separated by newlines or `;`."""
    configs = []
    for line in re.split(r"[;\n]", text):
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 3:
            raise ValueError(f"Expected 'model | weights | batch', got {line!r}")
        model, weights, batch = parts
        if weights not in ("16bit", "bnb-8bit", "bnb-4bit"):
            raise ValueError(f"Unknown weights {weights!r}")
        configs.append({"model": model, "revision": default_revisions.get(model),
                        "weights": weights, "batch_size": int(batch)})
    return configs


def pick_config(rows: list[dict], suite: str, max_drop: float, agreement_floor: float | None = None):
    """Fastest config within `max_drop` of the best balanced accuracy in the sweep.

    `agreement_floor` also requires argmax agreement with the committed BF16
    reference; leave it None when comparing different base models.
    """
    candidates = [row for row in rows if row["suite"] == suite]
    if not candidates:
        return None
    best = max(row["bal_acc"] for row in candidates)
    kept = [
        row for row in candidates
        if row["bal_acc"] >= best - max_drop
        and (agreement_floor is None or (row["argmax_agree"] or 0) >= agreement_floor)
    ]
    return max(kept, key=lambda row: row["dec_per_s"]) if kept else None


def release_cuda() -> None:
    """Return cached CUDA memory; `del` the model and tokenizer in the caller first."""
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def write_create_only(path: Path, payload) -> Path:
    """Benchmark outputs are create-only; never replace an earlier run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        if isinstance(payload, list):
            for row in payload:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        else:
            json.dump(payload, stream, indent=1, allow_nan=False, default=str)
    return path

