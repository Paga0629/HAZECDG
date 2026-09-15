#!/usr/bin/env python3
"""
Paper evaluator for Haze Counter-Degradation Guidance (HazeCDG).

This script follows the evaluation protocol used in the accompanying ICLR 2027
paper.  It evaluates only the three real-world haze benchmarks reported in the
paper and only the five no-reference metrics reported in the main experiments:

    FADE      (lower is better)
    Q-Align   (higher is better)
    CLIPIQA   (higher is better)
    MUSIQ     (higher is better)
    BRISQUE   (lower is better)

Supported datasets
------------------
The public repository uses a flat dataset layout:

    dataset/RTTS/
    dataset/URHI/
    dataset/Fattal/

To reproduce the paper, place the 4,322 RTTS images, the paper's 150-image URHI
subset, and the 31 Fattal images in the corresponding directories.

Supported output layouts
------------------------
LHD / DiffDehaze:

    outputs/LHD/baseline/<DATASET>/
    outputs/LHD/hazecdg/<DATASET>/K005/
    outputs/LHD/hazecdg/<DATASET>/K010/
    outputs/LHD/hazecdg/<DATASET>/K025/
    outputs/LHD/hazecdg/<DATASET>/K050/
    outputs/LHD/dctta/<DATASET>/

DOD:

    outputs/DOD/baseline/<DATASET>/
    outputs/DOD/hazecdg/<DATASET>/

External comparison methods can be evaluated with the same implementation by
setting ``evaluation.model_name`` in their YAML.  Their default output layout
is

    outputs/<MODEL_NAME>/<DATASET>/

An explicit ``data.output_root`` always overrides the automatic output path.

Examples
--------
Evaluate the output selected by an LHD YAML:

    python scripts/eval_metrics.py --config configs/eval_LHD_HazeCDG.yaml

Evaluate DOD output:

    python scripts/eval_metrics.py --config configs/eval_DOD_HazeCDG.yaml

Evaluate the original dataset inputs as a metric sanity check:

    python scripts/eval_metrics.py \
        --config configs/eval_LHD_HazeCDG.yaml \
        --source input

Evaluate a subset of the five paper metrics:

    python scripts/eval_metrics.py \
        --config configs/eval_LHD_HazeCDG.yaml \
        --metrics fade,qalign,clipiqa

Metric implementation notes
---------------------------
- PyIQA is used for Q-Align, CLIPIQA, MUSIQ, and BRISQUE.
- The project environment uses pyiqa==0.1.13.
- FADE is not part of PyIQA.  The official MATLAB implementation is preferred;
  PyFADE (fade-python) is supported as a fallback.
- RTTS, URHI, and Fattal are evaluated without paired clean references in this
  work, so PSNR, SSIM, and LPIPS are intentionally not part of this evaluator.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import yaml


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_OUTPUT_NAME = "LHD"
DOD_OUTPUT_NAME = "DOD"
MODE_AWARE_MODEL_FAMILIES = {MODEL_OUTPUT_NAME, DOD_OUTPUT_NAME}
SUPPORTED_MODES = ("baseline", "hazecdg", "dctta")
RECOMMENDED_PYIQA_VERSION = "0.1.13"

# Exact no-reference metric set reported in the paper.
PAPER_METRICS = ["FADE", "Q-Align", "CLIPIQA", "MUSIQ", "BRISQUE"]

DATASET_ALIASES = {
    "rtts": "rtts",
    "urhi": "urhi",
    "fattal": "fattal",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    output_name: str
    input_relative_dir: Path
    expected_count: int


DATASETS: Dict[str, DatasetSpec] = {
    "rtts": DatasetSpec(
        key="rtts",
        output_name="RTTS",
        input_relative_dir=Path("RTTS"),
        expected_count=4322,
    ),
    "urhi": DatasetSpec(
        key="urhi",
        output_name="URHI",
        input_relative_dir=Path("URHI"),
        expected_count=150,
    ),
    "fattal": DatasetSpec(
        key="fattal",
        output_name="Fattal",
        input_relative_dir=Path("Fattal"),
        expected_count=31,
    ),
}


@dataclass(frozen=True)
class MetricSpec:
    label: str
    backend: str
    direction: str  # "higher" | "lower"
    kind: str       # "pyiqa" | "fade"


@dataclass(frozen=True)
class EvaluationTarget:
    project_root: Path
    model_name: str
    mode: str
    dataset: str
    dataset_name: str
    source: str
    label: str
    image_root: Path
    metrics_root: Path
    k: Optional[int] = None


def build_metric_profile(
    brisque_variant: str = "brisque_matlab",
) -> Dict[str, MetricSpec]:
    if brisque_variant not in {"brisque", "brisque_matlab"}:
        raise ValueError("brisque_variant must be 'brisque' or 'brisque_matlab'.")
    return {
        "FADE": MetricSpec("FADE", "fade", "lower", "fade"),
        "Q-Align": MetricSpec("Q-Align", "qalign", "higher", "pyiqa"),
        "CLIPIQA": MetricSpec("CLIPIQA", "clipiqa", "higher", "pyiqa"),
        "MUSIQ": MetricSpec("MUSIQ", "musiq", "higher", "pyiqa"),
        "BRISQUE": MetricSpec("BRISQUE", brisque_variant, "lower", "pyiqa"),
    }


# =============================================================================
# Configuration / path helpers
# =============================================================================


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")
    return cfg


def nested_get(d: Mapping[str, Any], keys: Sequence[str], default=None):
    cur: Any = d
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def project_root_from_script() -> Path:
    # scripts/eval_metrics.py -> repository root
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(str(value)).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def normalize_dataset_name(name: str) -> str:
    key = str(name or "").strip().lower()
    if key not in DATASET_ALIASES:
        supported = ", ".join(spec.output_name for spec in DATASETS.values())
        raise ValueError(f"Unsupported data.dataset={name!r}. Supported: {supported}")
    return DATASET_ALIASES[key]


def normalize_mode(mode: str) -> str:
    key = str(mode or "").strip().lower()
    if key not in SUPPORTED_MODES:
        raise ValueError(
            "experiment.mode must be one of: "
            + ", ".join(repr(x) for x in SUPPORTED_MODES)
        )
    return key


def list_images(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p.resolve()
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def resolve_dataset_root(project_root: Path, cfg: Mapping[str, Any]) -> Path:
    return resolve_path(
        project_root,
        nested_get(cfg, ["data", "dataset_root"], "./dataset"),
    )


def resolve_dataset_input_root(
    project_root: Path,
    cfg: Mapping[str, Any],
    dataset: str,
) -> Path:
    explicit = str(nested_get(cfg, ["data", "input_root"], "") or "").strip()
    if explicit:
        path = resolve_path(project_root, explicit)
    else:
        dataset_root = resolve_dataset_root(project_root, cfg)
        path = (dataset_root / DATASETS[dataset].input_relative_dir).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset input directory does not exist: {path}")
    return path


def detect_model_output_name(
    cfg: Mapping[str, Any],
    config_path: Optional[Path] = None,
) -> str:
    """Detect the model family used to resolve the output directory.

    Preferred for external comparison methods:

        evaluation:
          model_name: HNDiff

    LHD remains the default.  DOD is detected from its path keys.
    """
    explicit = str(nested_get(cfg, ["evaluation", "model_name"], "") or "").strip()
    if explicit:
        return explicit

    dod_path_keys = (
        "dod_repo_root",
        "dod_sd21_root",
        "dod_stage1_checkpoint",
        "dod_stage2_checkpoint",
        "dod_mfm_checkpoint",
        "dod_prompt_embeds",
    )
    if any(
        str(nested_get(cfg, ["paths", key], "") or "").strip()
        for key in dod_path_keys
    ):
        return DOD_OUTPUT_NAME

    # Convenience detection for comparison-method YAMLs already used in the
    # project.  evaluation.model_name is still preferred for public configs.
    hints = [
        str(config_path or ""),
        str(nested_get(cfg, ["paths", "repo_root"], "") or ""),
        str(nested_get(cfg, ["paths", "checkpoint"], "") or ""),
        str(nested_get(cfg, ["paths", "model_id"], "") or ""),
        str(nested_get(cfg, ["paths", "base_checkpoint"], "") or ""),
        str(nested_get(cfg, ["paths", "lora_checkpoint"], "") or ""),
    ]
    joined = " ".join(hints).lower()
    external_names = {
        "hazeflow": "HazeFlow",
        "bilalora": "BiLaLoRA",
        "dehazesb": "DehazeSB",
        "hndiff": "HNDiff",
        "pdda": "PDDA",
    }
    for token, name in external_names.items():
        if token in joined:
            return name

    # CoA is short and ambiguous; detect it only from explicit path/name forms.
    if "eval_coa" in joined or "/coa/" in joined or "\\coa\\" in joined:
        return "CoA"

    return MODEL_OUTPUT_NAME


def model_uses_mode_layout(model_name: str) -> bool:
    return str(model_name) in MODE_AWARE_MODEL_FAMILIES


def normalize_model_mode(model_name: str, mode: str) -> str:
    key = normalize_mode(mode)
    if model_name == DOD_OUTPUT_NAME and key not in {"baseline", "hazecdg"}:
        raise ValueError("DOD evaluation supports baseline or hazecdg only.")
    return key


def resolve_output_root(
    project_root: Path,
    cfg: Mapping[str, Any],
    mode: str,
    dataset: str,
    model_name: str = MODEL_OUTPUT_NAME,
) -> Path:
    explicit = str(nested_get(cfg, ["data", "output_root"], "") or "").strip()
    if explicit:
        return resolve_path(project_root, explicit)

    dataset_name = DATASETS[dataset].output_name
    if model_uses_mode_layout(model_name):
        return (
            project_root / "outputs" / model_name / mode / dataset_name
        ).resolve()
    return (project_root / "outputs" / model_name / dataset_name).resolve()


def resolve_evaluation_target(
    project_root: Path,
    cfg: Mapping[str, Any],
    source: str = "output",
    metrics_root_override: str = "",
    mode_override: str = "",
    config_path: Optional[Path] = None,
) -> EvaluationTarget:
    source = str(source).strip().lower()
    if source not in {"output", "input"}:
        raise ValueError("source must be either 'output' or 'input'.")

    model_name = detect_model_output_name(cfg, config_path=config_path)
    if model_uses_mode_layout(model_name):
        mode_raw = mode_override or nested_get(cfg, ["experiment", "mode"], "baseline")
        mode = normalize_model_mode(model_name, mode_raw)
    else:
        if mode_override:
            raise ValueError(
                f"--mode is supported only for LHD/DOD outputs, not {model_name}."
            )
        mode = model_name.lower()

    dataset = normalize_dataset_name(nested_get(cfg, ["data", "dataset"], "RTTS"))
    dataset_name = DATASETS[dataset].output_name

    if source == "output":
        image_root = resolve_output_root(
            project_root, cfg, mode, dataset, model_name=model_name
        )
        default_metrics_root = image_root.parent / f"{image_root.name}_metrics"
        label = mode if model_uses_mode_layout(model_name) else model_name
    else:
        image_root = resolve_dataset_input_root(project_root, cfg, dataset)
        default_metrics_root = (
            project_root
            / "outputs"
            / model_name
            / "input"
            / f"{dataset_name}_metrics"
        ).resolve()
        label = "input"

    metrics_root = (
        resolve_path(project_root, metrics_root_override)
        if str(metrics_root_override or "").strip()
        else default_metrics_root.resolve()
    )

    return EvaluationTarget(
        project_root=project_root.resolve(),
        model_name=model_name,
        mode=mode,
        dataset=dataset,
        dataset_name=dataset_name,
        source=source,
        label=label,
        image_root=image_root.resolve(),
        metrics_root=metrics_root,
    )


# =============================================================================
# LHD sparse-HazeCDG multi-K output handling
# =============================================================================


def parse_hazecdg_anchor_counts(cfg: Mapping[str, Any]) -> List[int]:
    raw = nested_get(cfg, ["hazecdg", "anchor_counts"], None)
    if raw is None:
        raise ValueError("hazecdg.anchor_counts is required for LHD HazeCDG evaluation.")
    if isinstance(raw, (int, float)):
        raw = [int(raw)]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("hazecdg.anchor_counts must be a sequence of positive integers.")
    ks = sorted({int(x) for x in raw})
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("hazecdg.anchor_counts must contain positive integers.")
    return ks


def resolve_hazecdg_multik_targets(
    project_root: Path,
    cfg: Mapping[str, Any],
    metrics_root_override: str = "",
    model_name: str = MODEL_OUTPUT_NAME,
) -> List[EvaluationTarget]:
    dataset = normalize_dataset_name(nested_get(cfg, ["data", "dataset"], "RTTS"))
    dataset_name = DATASETS[dataset].output_name
    base_output = resolve_output_root(
        project_root, cfg, "hazecdg", dataset, model_name=model_name
    )
    ks = parse_hazecdg_anchor_counts(cfg)

    custom_metrics_base = (
        resolve_path(project_root, metrics_root_override)
        if str(metrics_root_override or "").strip()
        else None
    )

    targets: List[EvaluationTarget] = []
    for k in ks:
        k_name = f"K{k:03d}"
        image_root = (base_output / k_name).resolve()
        metrics_root = (
            (custom_metrics_base / k_name).resolve()
            if custom_metrics_base is not None
            else (image_root / "metrics").resolve()
        )
        targets.append(
            EvaluationTarget(
                project_root=project_root.resolve(),
                model_name=model_name,
                mode="hazecdg",
                dataset=dataset,
                dataset_name=dataset_name,
                source="output",
                label=f"HazeCDG K={k}",
                image_root=image_root,
                metrics_root=metrics_root,
                k=int(k),
            )
        )
    return targets


def resolve_inference_step_count(cfg: Mapping[str, Any], model_name: str) -> int:
    if model_name == DOD_OUTPUT_NAME:
        raw = nested_get(cfg, ["inference", "steps"], 1)
    else:
        raw = nested_get(cfg, ["inference", "steps"], 50)
    steps = int(raw)
    if steps <= 0:
        raise ValueError(f"Invalid inference step count for {model_name}: {steps}.")
    return steps


def write_hazecdg_comparison(
    base_output_root: Path,
    results: Sequence[Mapping[str, Any]],
    profile: Mapping[str, MetricSpec],
    model_name: str = MODEL_OUTPUT_NAME,
) -> None:
    if not results:
        return
    base_output_root.mkdir(parents=True, exist_ok=True)

    metric_labels = [
        metric
        for metric in PAPER_METRICS
        if any(metric in (result.get("summary") or {}) for result in results)
    ]

    rows: List[Dict[str, Any]] = []
    for result in sorted(results, key=lambda x: int(x["K"])):
        row: Dict[str, Any] = {
            "K": int(result["K"]),
            "sensing_ratio": float(result.get("sensing_ratio", float("nan"))),
            "images": int(result.get("images", 0)),
        }
        summary = result.get("summary") or {}
        for metric in metric_labels:
            stats = summary.get(metric)
            row[metric] = "" if not stats else float(stats["mean"])
        rows.append(row)

    csv_path = base_output_root / "comparison.csv"
    fields = ["K", "sensing_ratio", "images"] + metric_labels
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "model": model_name,
        "mode": "hazecdg",
        "results": rows,
        "metric_directions": {
            metric: profile[metric].direction for metric in metric_labels
        },
    }
    (base_output_root / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    width = max(104, 22 + 12 * len(metric_labels))
    lines = [
        "=" * width,
        f"{model_name} / HazeCDG — SENSING-BUDGET COMPARISON",
        "=" * width,
    ]
    header = f"{'K':>5} {'Sense':>8} {'N':>6}"
    for metric in metric_labels:
        arrow = "↑" if profile[metric].direction == "higher" else "↓"
        header += f" {metric + arrow:>11}"
    lines.append(header)
    lines.append("-" * width)

    for row in rows:
        sense = row["sensing_ratio"]
        sense_text = "-" if not np.isfinite(sense) else f"{100.0 * sense:.1f}%"
        line = f"{int(row['K']):>5d} {sense_text:>8} {int(row['images']):>6d}"
        for metric in metric_labels:
            value = row.get(metric, "")
            line += f" {('-' if value == '' else f'{float(value):.6f}'):>11}"
        lines.append(line)

    lines.extend(
        [
            "=" * width,
            "Direction: FADE↓, BRISQUE↓; Q-Align↑, CLIPIQA↑, MUSIQ↑.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (base_output_root / "comparison.txt").write_text(text, encoding="utf-8")
    print("\n" + text, end="")
    print(f"Comparison CSV : {csv_path}")


# =============================================================================
# Metric selection / persistence
# =============================================================================


def safe_package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def scalarize(value: Any) -> float:
    try:
        import torch

        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"Metric returned {value.numel()} values, expected one.")
            return float(value.detach().float().cpu().item())
    except ImportError:
        pass
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Metric returned shape {arr.shape}, expected scalar.")
    return float(arr.reshape(-1)[0])


def parse_metric_selection(raw: str) -> List[str]:
    aliases = {
        "fade": "FADE",
        "qalign": "Q-Align",
        "q-align": "Q-Align",
        "clipiqa": "CLIPIQA",
        "clip-iqa": "CLIPIQA",
        "musiq": "MUSIQ",
        "brisque": "BRISQUE",
    }
    text = str(raw or "").strip()
    if not text or text.lower() == "all":
        return list(PAPER_METRICS)

    selected: List[str] = []
    for token in text.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(
                f"Unknown metric token: {token!r}. "
                "Supported: fade, qalign, clipiqa, musiq, brisque."
            )
        label = aliases[key]
        if label not in selected:
            selected.append(label)
    if not selected:
        raise ValueError("No metrics selected.")
    return selected


def read_existing_csv(
    csv_path: Path,
    image_keys: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {key: {} for key in image_keys}
    if not csv_path.is_file():
        return rows
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("image", "")
            if key not in rows:
                continue
            for metric in PAPER_METRICS:
                raw = row.get(metric, "")
                if raw not in {None, ""}:
                    try:
                        rows[key][metric] = float(raw)
                    except ValueError:
                        pass
    return rows


def write_per_image_csv(
    csv_path: Path,
    image_keys: Sequence[str],
    scores: Mapping[str, Mapping[str, float]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["image"] + PAPER_METRICS
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for key in image_keys:
            row: Dict[str, Any] = {"image": key}
            for metric in PAPER_METRICS:
                value = scores.get(key, {}).get(metric)
                row[metric] = "" if value is None else f"{float(value):.10f}"
            writer.writerow(row)
    os.replace(tmp_path, csv_path)


def metric_is_complete(
    metric_label: str,
    image_keys: Sequence[str],
    scores: Mapping[str, Mapping[str, float]],
) -> bool:
    return bool(image_keys) and all(
        metric_label in scores.get(key, {}) for key in image_keys
    )


def _progress(metric: str, idx: int, total: int, path: Path, score: float) -> None:
    if idx == 1 or idx == total or idx % 25 == 0:
        print(f"[{metric:8s}] {idx:04d}/{total:04d} {path.name} -> {score:.6f}")


# =============================================================================
# Metric backends
# =============================================================================


def evaluate_pyiqa_metric(
    spec: MetricSpec,
    image_paths: Sequence[Path],
    image_keys: Sequence[str],
    scores: Dict[str, Dict[str, float]],
    device: str,
    csv_path: Path,
) -> None:
    try:
        import torch
        import pyiqa
    except Exception as exc:
        raise RuntimeError(
            "PyIQA metrics require pyiqa. The project environment uses "
            f"pyiqa=={RECOMMENDED_PYIQA_VERSION}."
        ) from exc

    print(f"\n[metric] {spec.label} <- PyIQA '{spec.backend}' on {device}")
    model = pyiqa.create_metric(spec.backend, device=device)
    model.eval()

    with torch.inference_mode():
        for idx, (path, key) in enumerate(zip(image_paths, image_keys), 1):
            kwargs = {"task_": "quality"} if spec.backend.startswith("qalign") else {}
            value = model(str(path), **kwargs)
            score = scalarize(value)
            if not math.isfinite(score):
                raise RuntimeError(f"Non-finite {spec.label} score for {path}: {score}")
            scores[key][spec.label] = score
            _progress(spec.label, idx, len(image_paths), path, score)
            if idx % 25 == 0:
                write_per_image_csv(csv_path, image_keys, scores)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_per_image_csv(csv_path, image_keys, scores)


def choose_fade_backend(
    requested: str,
    fade_matlab_dir: Optional[Path],
    matlab_cmd: str,
) -> str:
    requested = requested.strip().lower()
    if requested not in {"auto", "matlab", "pyfade"}:
        raise ValueError("fade_backend must be auto | matlab | pyfade")

    has_matlab_fade = bool(
        fade_matlab_dir is not None
        and (fade_matlab_dir / "FADE.m").is_file()
        and shutil.which(matlab_cmd)
    )
    has_pyfade = importlib.util.find_spec("pyfade") is not None

    if requested == "matlab":
        if not has_matlab_fade:
            raise RuntimeError(
                "FADE MATLAB backend requested, but MATLAB and/or FADE.m was not found."
            )
        return "matlab"
    if requested == "pyfade":
        if not has_pyfade:
            raise RuntimeError("PyFADE is not installed. Install package 'fade-python'.")
        return "pyfade"
    if has_matlab_fade:
        return "matlab"
    if has_pyfade:
        return "pyfade"
    raise RuntimeError(
        "FADE backend not available. Install the official MATLAB FADE code or "
        "the 'fade-python' package."
    )


def _matlab_quote(path: Path) -> str:
    return str(path.resolve()).replace("'", "''").replace("\\", "/")


def evaluate_fade_matlab(
    image_paths: Sequence[Path],
    image_keys: Sequence[str],
    scores: Dict[str, Dict[str, float]],
    fade_dir: Path,
    matlab_cmd: str,
    csv_path: Path,
) -> None:
    print(f"\n[metric] FADE <- official MATLAB implementation: {fade_dir}")
    with tempfile.TemporaryDirectory(prefix="hazecdg_fade_") as td:
        td_path = Path(td)
        image_list = td_path / "images.txt"
        score_file = td_path / "scores.txt"
        script_file = td_path / "run_fade.m"

        image_list.write_text(
            "\n".join(str(path.resolve()) for path in image_paths) + "\n",
            encoding="utf-8",
        )
        script = f"""
addpath(genpath('{_matlab_quote(fade_dir)}'));
fid_in = fopen('{_matlab_quote(image_list)}', 'r');
fid_out = fopen('{_matlab_quote(score_file)}', 'w');
if fid_in < 0 || fid_out < 0
    error('Could not open FADE temporary files.');
end
idx = 0;
while true
    p = fgetl(fid_in);
    if ~ischar(p), break; end
    if isempty(p), continue; end
    idx = idx + 1;
    I = imread(p);
    D = FADE(I);
    fprintf(fid_out, '%d,%.12f\\n', idx, double(D));
end
fclose(fid_in);
fclose(fid_out);
"""
        script_file.write_text(script, encoding="utf-8")
        subprocess.run(
            [matlab_cmd, "-batch", f"run('{_matlab_quote(script_file)}')"],
            check=True,
        )

        lines = score_file.read_text(encoding="utf-8").splitlines()
        if len(lines) != len(image_paths):
            raise RuntimeError(
                f"MATLAB FADE returned {len(lines)} scores for "
                f"{len(image_paths)} images."
            )
        for idx, (line, path, key) in enumerate(
            zip(lines, image_paths, image_keys), 1
        ):
            fields = line.split(",")
            if len(fields) != 2:
                raise RuntimeError(f"Malformed MATLAB FADE output: {line}")
            score = float(fields[1])
            if not math.isfinite(score):
                raise RuntimeError(f"Non-finite FADE score for {path}: {score}")
            scores[key]["FADE"] = score
            _progress("FADE", idx, len(image_paths), path, score)

    write_per_image_csv(csv_path, image_keys, scores)


def evaluate_fade_pyfade(
    image_paths: Sequence[Path],
    image_keys: Sequence[str],
    scores: Dict[str, Dict[str, float]],
    csv_path: Path,
) -> None:
    try:
        from pyfade import fade
    except Exception as exc:
        raise RuntimeError(
            "Could not import PyFADE. Install package 'fade-python'."
        ) from exc

    print("\n[metric] FADE <- PyFADE fallback")
    for idx, (path, key) in enumerate(zip(image_paths, image_keys), 1):
        score = float(fade(str(path)))
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite FADE score for {path}: {score}")
        scores[key]["FADE"] = score
        _progress("FADE", idx, len(image_paths), path, score)
        if idx % 25 == 0:
            write_per_image_csv(csv_path, image_keys, scores)
    write_per_image_csv(csv_path, image_keys, scores)


# =============================================================================
# Summary
# =============================================================================


def compute_summary(
    image_keys: Sequence[str],
    scores: Mapping[str, Mapping[str, float]],
    metric_labels: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for metric in metric_labels:
        values = [
            float(scores[key][metric])
            for key in image_keys
            if metric in scores.get(key, {})
        ]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        result[metric] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": int(arr.size),
        }
    return result


def write_summary_files(
    target: EvaluationTarget,
    profile: Mapping[str, MetricSpec],
    selected_metrics: Sequence[str],
    image_count: int,
    summary: Mapping[str, Mapping[str, float]],
    fade_backend: Optional[str],
    pyiqa_version: Optional[str],
    elapsed_seconds: float,
) -> None:
    target.metrics_root.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model": target.model_name,
        "source": target.source,
        "mode": target.mode,
        "dataset": target.dataset_name,
        "image_root": str(target.image_root),
        "image_count": image_count,
        "expected_paper_count": DATASETS[target.dataset].expected_count,
        "selected_metrics": list(selected_metrics),
        "metrics": summary,
        "metric_backends": {
            label: profile[label].backend for label in selected_metrics
        },
        "fade_backend": fade_backend,
        "pyiqa_version": pyiqa_version,
        "recommended_pyiqa_version": RECOMMENDED_PYIQA_VERSION,
        "elapsed_seconds": elapsed_seconds,
    }

    json_path = target.metrics_root / "summary.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    width = 92
    lines = [
        "=" * width,
        f"{target.model_name} — PAPER METRIC SUMMARY",
        "=" * width,
        f"Model        : {target.model_name}",
        f"Source       : {target.source}",
        f"Mode         : {target.mode}",
        f"Dataset      : {target.dataset_name}",
        f"Image root   : {target.image_root}",
        f"Images       : {image_count}",
        f"Paper count  : {DATASETS[target.dataset].expected_count}",
        f"Metrics      : {', '.join(selected_metrics)}",
        f"PyIQA        : {pyiqa_version or 'not used'}",
        f"FADE backend : {fade_backend or 'not used'}",
        "-" * width,
        f"{'Metric':<12} {'Direction':<12} {'Mean':>14} {'Std':>14} {'Count':>8}",
        "-" * width,
    ]

    for label in selected_metrics:
        if label not in summary:
            continue
        spec = profile[label]
        stats = summary[label]
        arrow = "↑" if spec.direction == "higher" else "↓"
        lines.append(
            f"{label:<12} {(arrow + ' ' + spec.direction):<12} "
            f"{stats['mean']:>14.6f} {stats['std']:>14.6f} "
            f"{int(stats['count']):>8d}"
        )

    lines.append("=" * width)
    expected = DATASETS[target.dataset].expected_count
    if image_count != expected:
        lines.append(
            f"WARNING: paper protocol uses {expected} {target.dataset_name} images; "
            f"this run used {image_count}."
        )
    if pyiqa_version and pyiqa_version != RECOMMENDED_PYIQA_VERSION:
        lines.append(
            f"WARNING: installed pyiqa={pyiqa_version}; the project environment "
            f"uses pyiqa=={RECOMMENDED_PYIQA_VERSION}."
        )
    if fade_backend == "pyfade":
        lines.append(
            "NOTE: PyFADE is a fallback implementation; use the official MATLAB "
            "FADE implementation for the preferred reproduction path."
        )

    text = "\n".join(lines) + "\n"
    (target.metrics_root / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text, end="")
    print(f"CSV          : {target.metrics_root / 'per_image_metrics.csv'}")
    print(f"JSON         : {json_path}")


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_one_target(
    *,
    project_root: Path,
    config_path: Path,
    cfg: Mapping[str, Any],
    target: EvaluationTarget,
    args,
) -> Dict[str, Any]:
    if not target.image_root.is_dir():
        raise FileNotFoundError(
            f"Selected image directory does not exist: {target.image_root}\n"
            "Run inference first, or use --source input."
        )

    image_paths = list_images(target.image_root)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No images found under: {target.image_root}")

    image_keys = [
        path.relative_to(target.image_root).as_posix() for path in image_paths
    ]
    if len(set(image_keys)) != len(image_keys):
        raise RuntimeError("Duplicate relative image paths detected.")

    selected = parse_metric_selection(args.metrics)
    profile = build_metric_profile(args.brisque_variant)
    device = args.device.strip() or str(
        nested_get(cfg, ["runtime", "device"], "cuda:0")
    )

    target.metrics_root.mkdir(parents=True, exist_ok=True)
    csv_path = target.metrics_root / "per_image_metrics.csv"
    scores = read_existing_csv(csv_path, image_keys)
    if args.force:
        for key in image_keys:
            for metric in selected:
                scores[key].pop(metric, None)

    expected = DATASETS[target.dataset].expected_count
    print("=" * 104)
    print("HazeCDG paper metric evaluation")
    print("=" * 104)
    print(f"Config       : {config_path}")
    print(f"Model        : {target.model_name}")
    print(f"Source       : {target.source}")
    print(f"Mode         : {target.mode}")
    if target.k is not None:
        print(f"HazeCDG K    : {target.k}")
    print(f"Dataset      : {target.dataset_name}")
    print(f"Image root   : {target.image_root}")
    print(f"Images       : {len(image_paths)} (paper protocol: {expected})")
    print(f"Metrics      : {', '.join(selected)}")
    print(f"Metric output: {target.metrics_root}")
    print(f"Device       : {device}")
    print("=" * 104)

    if len(image_paths) != expected and args.max_images == 0:
        print(
            f"[warn] {target.dataset_name}: expected {expected} images for paper "
            f"reproduction, found {len(image_paths)}."
        )

    pyiqa_version = safe_package_version("pyiqa")
    if any(profile[metric].kind == "pyiqa" for metric in selected):
        if pyiqa_version is None:
            raise RuntimeError(
                "pyiqa is not installed. The project environment uses "
                f"pyiqa=={RECOMMENDED_PYIQA_VERSION}."
            )
        print(f"[env] pyiqa version : {pyiqa_version}")
        if pyiqa_version != RECOMMENDED_PYIQA_VERSION:
            print(
                f"[warn] project environment uses pyiqa=={RECOMMENDED_PYIQA_VERSION}; "
                f"installed={pyiqa_version}."
            )

    fade_backend_used: Optional[str] = None
    start = time.time()

    for label in selected:
        spec = profile[label]
        if not args.force and metric_is_complete(label, image_keys, scores):
            print(f"[skip] {label}: complete values already exist in {csv_path}")
            continue

        if spec.kind == "fade":
            fade_dir = resolve_path(project_root, args.fade_matlab_dir)
            fade_backend_used = choose_fade_backend(
                args.fade_backend, fade_dir, args.matlab_cmd
            )
            if fade_backend_used == "matlab":
                evaluate_fade_matlab(
                    image_paths,
                    image_keys,
                    scores,
                    fade_dir,
                    args.matlab_cmd,
                    csv_path,
                )
            else:
                evaluate_fade_pyfade(
                    image_paths, image_keys, scores, csv_path
                )
        elif spec.kind == "pyiqa":
            evaluate_pyiqa_metric(
                spec, image_paths, image_keys, scores, device, csv_path
            )
        else:
            raise RuntimeError(f"Unsupported metric kind: {spec.kind}")

    write_per_image_csv(csv_path, image_keys, scores)
    summary = compute_summary(image_keys, scores, selected)
    elapsed = time.time() - start
    write_summary_files(
        target=target,
        profile=profile,
        selected_metrics=selected,
        image_count=len(image_paths),
        summary=summary,
        fade_backend=fade_backend_used,
        pyiqa_version=pyiqa_version,
        elapsed_seconds=elapsed,
    )
    return {
        "target": target,
        "summary": summary,
        "selected_metrics": selected,
        "profile": profile,
        "images": len(image_paths),
    }


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default="./configs/eval_LHD_HazeCDG.yaml")
    parser.add_argument(
        "--source",
        choices=["output", "input"],
        default="output",
        help="Evaluate model outputs or the original dataset inputs.",
    )
    parser.add_argument(
        "--mode",
        choices=list(SUPPORTED_MODES),
        default="",
        help="Optional LHD/DOD mode override.",
    )
    parser.add_argument(
        "--metrics",
        default="all",
        help=(
            "Comma-separated subset of the five paper metrics, or 'all': "
            "fade,qalign,clipiqa,musiq,brisque."
        ),
    )
    parser.add_argument(
        "--device",
        default="",
        help="Metric device; empty uses runtime.device from YAML.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="0 means all images found; positive N evaluates the first N sorted images.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute selected metrics even if complete values already exist.",
    )
    parser.add_argument(
        "--metrics-root",
        default="",
        help=(
            "Optional custom metric output directory. For LHD multi-K HazeCDG, "
            "K005/K010/... subdirectories are created underneath it."
        ),
    )
    parser.add_argument(
        "--brisque-variant",
        choices=["brisque_matlab", "brisque"],
        default="brisque_matlab",
    )
    parser.add_argument(
        "--fade-backend",
        choices=["auto", "matlab", "pyfade"],
        default="auto",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["RTTS", "URHI", "Fattal"],
        default="",
        help="Override data.dataset from YAML.",
    )

    parser.add_argument("--fade-matlab-dir", default="./third_party/FADE")
    parser.add_argument("--matlab-cmd", default="matlab")
    args = parser.parse_args()

    if args.max_images < 0:
        raise ValueError("--max-images must be >= 0.")

    project_root = project_root_from_script()
    config_path = resolve_path(project_root, args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = load_yaml(config_path)
    if args.dataset:
        cfg.setdefault("data", {})
        cfg["data"]["dataset"] = args.dataset
    model_name = detect_model_output_name(cfg, config_path=config_path)
    if model_uses_mode_layout(model_name):
        requested_mode = normalize_model_mode(
            model_name,
            args.mode or nested_get(cfg, ["experiment", "mode"], "baseline"),
        )
    else:
        if args.mode:
            raise ValueError(
                f"--mode is supported only for LHD/DOD outputs, not {model_name}."
            )
        requested_mode = model_name.lower()

    # LHD HazeCDG always writes Kxxx directories, including the default K=5
    # setting.  Evaluate each K requested in the inference YAML and then write
    # a compact cross-budget comparison table.
    is_lhd_multik_hazecdg = (
        model_name == MODEL_OUTPUT_NAME
        and args.source == "output"
        and requested_mode == "hazecdg"
        and nested_get(cfg, ["hazecdg", "anchor_counts"], None) is not None
    )

    if is_lhd_multik_hazecdg:
        targets = resolve_hazecdg_multik_targets(
            project_root,
            cfg,
            metrics_root_override=args.metrics_root,
            model_name=model_name,
        )
        base_output = resolve_output_root(
            project_root,
            cfg,
            "hazecdg",
            targets[0].dataset,
            model_name=model_name,
        )
        reference_steps = resolve_inference_step_count(cfg, model_name)

        print("=" * 104)
        print("LHD HazeCDG sensing-budget evaluation")
        print("=" * 104)
        print(f"Dataset      : {targets[0].dataset_name}")
        print(f"K values     : {[target.k for target in targets]}")
        print(f"Base output  : {base_output}")
        print(f"Reverse steps: {reference_steps}")
        print(f"Metrics      : {', '.join(parse_metric_selection(args.metrics))}")
        print("=" * 104)

        results: List[Dict[str, Any]] = []
        for target in targets:
            assert target.k is not None
            sensing_ratio = float(target.k) / float(reference_steps)
            print("\n" + "#" * 104)
            print(
                f"# HazeCDG K={target.k} "
                f"({100.0 * sensing_ratio:.1f}% of {reference_steps} reverse steps)"
            )
            print("#" * 104)
            out = evaluate_one_target(
                project_root=project_root,
                config_path=config_path,
                cfg=cfg,
                target=target,
                args=args,
            )
            results.append(
                {
                    "K": int(target.k),
                    "sensing_ratio": sensing_ratio,
                    "images": int(out["images"]),
                    "summary": out["summary"],
                }
            )

        profile = build_metric_profile(args.brisque_variant)
        write_hazecdg_comparison(
            base_output, results, profile, model_name=model_name
        )
        return

    target = resolve_evaluation_target(
        project_root,
        cfg,
        source=args.source,
        metrics_root_override=args.metrics_root,
        mode_override=args.mode,
        config_path=config_path,
    )
    evaluate_one_target(
        project_root=project_root,
        config_path=config_path,
        cfg=cfg,
        target=target,
        args=args,
    )


if __name__ == "__main__":
    main()
