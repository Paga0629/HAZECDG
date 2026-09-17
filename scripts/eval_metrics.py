#!/usr/bin/env python3
"""
Unified evaluator for Learning-Hazing-to-Dehazing experiments.

Supported method/output modes
-----------------------------
LHD family:
    baseline | dfg | hazecdg | dctta

LD-RPS family:
    baseline | hazecdg (auto-detected from eval_LDRPS_HazeCDG.yaml / paths.ld_rps_repo_root)

DOD family:
    baseline | hazecdg (auto-detected from eval_DOD_HazeCDG.yaml / DOD path keys)

External dehazing methods:
    BiLaLoRA / HNDiff (auto-detected from config/path hints)

HazeCDG ablation family:
    Dedicated HazeCDG ablation YAMLs are supported.
    Runtime-only experiments are not passed through IQA metrics.

HazeCDG multi-budget layout
---------------------------
When ``hazecdg.anchor_counts`` is present (e.g. [5,10,20,40]), one invocation
automatically evaluates K005/K010/K020/K040 and writes a comparison table.

Supported datasets
------------------
    RTTS          -> datasets/dehaze/test/RTTS
    Haze4K        -> datasets/dehaze/benchmarks/Haze4K/test/haze
    LIVE500Foggy  -> datasets/dehaze/benchmarks/LIVE500Foggy
    URHI          -> datasets/dehaze/train/hazegen/URHI
    Fattal        -> datasets/dehaze/benchmarks/Fattal
    HSTS          -> datasets/dehaze/benchmarks/HSTS/input

Evaluation policy
-----------------
1. RTTS has no paired clean GT in this project, so the default ``auto``
   profile evaluates the five selected no-reference metrics.
2. Haze4K automatically searches for a paired GT directory (gt/GT/clear/clean).
   When GT is available, PSNR / SSIM / LPIPS are added automatically.
3. LIVE500Foggy is treated as no-reference unless ``evaluation.gt_root`` or
   ``data.gt_root`` is explicitly provided in the YAML.
4. Fattal is treated as a no-reference real-world test benchmark by default.
5. URHI is HazeGen training data. The script allows diagnostic evaluation but
   prints a leakage warning; do not use it as a paper test benchmark.

Default metric profiles
-----------------------
``auto``
    All datasets: FADE, Q-Align, CLIPIQA, MUSIQ, BRISQUE.
    Add PSNR / SSIM / LPIPS automatically when paired GT exists.

``primary``
    FADE, Q-Align, CLIPIQA, MUSIQ
    + PSNR, SSIM, LPIPS when GT exists.

``paper``
    FADE, Q-Align, CLIPIQA, MUSIQ, BRISQUE
    + PSNR, SSIM, LPIPS when GT exists.

Examples
--------
Evaluate output selected by YAML:

    python scripts/eval_metrics.py --config configs/eval_LHD_HazeCDG.yaml

Evaluate original RTTS inputs for paper-metric sanity check:

    python scripts/eval_metrics.py \
        --config configs/eval_LHD_HazeCDG.yaml \
        --source input

Force final LHD-paper metric profile:

    python scripts/eval_metrics.py \
        --config configs/eval_LHD_HazeCDG.yaml \
        --metric-profile paper

Evaluate DCTTA without editing YAML:

    python scripts/eval_metrics.py \
        --config configs/eval_LHD_HazeCDG.yaml \
        --mode dctta

Evaluate BiLaLoRA outputs simply by switching YAML:

    python scripts/eval_metrics.py \
        --config configs/eval_BiLaLoRA.yaml

Explicit metric override:

    python scripts/eval_metrics.py \
        --config configs/eval_LHD_HazeCDG.yaml \
        --metrics fade,qalign,clipiqa,psnr,ssim,lpips

Important reproduction notes
----------------------------
- The public LHD repository does not release the exact Table-1 metric script.
- The paper's DiffDehaze row uses AccSamp (50 steps, tau=800, omega=600, s=0.1).
- For closest paper-time NR calibration, pyiqa==0.1.13 is recommended.
- FADE is not in PyIQA. Official MATLAB FADE is preferred; PyFADE is fallback.
- Full-reference metrics require one-to-one GT pairing for every evaluated image.
  Missing/ambiguous GT pairs are treated as errors rather than silently skipped.
- Output and GT spatial sizes must match by default. Use
  --resize-output-to-gt only when you deliberately want that protocol.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_OUTPUT_NAME = "LHD"
LDRPS_OUTPUT_NAME = "LDRPS"
DOD_OUTPUT_NAME = "DOD"
BILALORA_OUTPUT_NAME = "BiLaLoRA"
HNDIFF_OUTPUT_NAME = "HNDiff"
MODE_AWARE_MODEL_FAMILIES = {MODEL_OUTPUT_NAME, LDRPS_OUTPUT_NAME, DOD_OUTPUT_NAME}
MULTIK_HAZECDG_MODEL_FAMILIES = {MODEL_OUTPUT_NAME, LDRPS_OUTPUT_NAME}
EXPECTED_RTTS_COUNT = 4322
RECOMMENDED_PYIQA_VERSION = "0.1.13"

SUPPORTED_MODES = (
    "baseline",
    "dfg",
    "hazecdg",
    "dctta",
    "direct",
    "projection",
)

HAZECDG_ABLATION_MODES = (
    "fixed_strength",
    "neutral_counter",
    "hold_space",
    "runtime",
)


NR_PRIMARY_METRICS = ["FADE", "Q-Align", "CLIPIQA", "MUSIQ"]
NR_PAPER_METRICS = [
    "FADE",
    "Q-Align",
    "CLIPIQA",
    "MUSIQ",
    "BRISQUE",
]
FR_METRICS = ["PSNR", "SSIM", "LPIPS"]
ALL_METRICS = NR_PAPER_METRICS + FR_METRICS

DATASET_ALIASES = {
    "rtts": "rtts",
    "reside_rtts": "rtts",
    "reside-rtts": "rtts",
    "haze4k": "haze4k",
    "haze_4k": "haze4k",
    "haze-4k": "haze4k",
    "live500foggy": "live500foggy",
    "live500": "live500foggy",
    "live_500_foggy": "live500foggy",
    "500_foggy": "live500foggy",
    "urhi": "urhi",
    "reside_urhi": "urhi",
    "reside-urhi": "urhi",
    "fattal": "fattal",
    "fattal31": "fattal",
    "fattal_31": "fattal",
    "fattal-31": "fattal",
    "hsts": "hsts",
    "reside_hsts": "hsts",
    "reside-hsts": "hsts",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    output_name: str
    input_relative_dir: Path
    gt_relative_candidates: Tuple[Path, ...]
    role: str  # "test" | "train"


DATASETS: Dict[str, DatasetSpec] = {
    "rtts": DatasetSpec(
        key="rtts",
        output_name="RTTS",
        input_relative_dir=Path("test/RTTS"),
        gt_relative_candidates=(),
        role="test",
    ),
    "haze4k": DatasetSpec(
        key="haze4k",
        output_name="Haze4K",
        input_relative_dir=Path("benchmarks/Haze4K/test/haze"),
        gt_relative_candidates=(
            Path("benchmarks/Haze4K/test/gt"),
            Path("benchmarks/Haze4K/test/GT"),
            Path("benchmarks/Haze4K/test/clear"),
            Path("benchmarks/Haze4K/test/clean"),
        ),
        role="test",
    ),
    "live500foggy": DatasetSpec(
        key="live500foggy",
        output_name="LIVE500Foggy",
        input_relative_dir=Path("benchmarks/LIVE500Foggy"),
        gt_relative_candidates=(),
        role="test",
    ),
    "urhi": DatasetSpec(
        key="urhi",
        output_name="URHI",
        input_relative_dir=Path("train/hazegen/URHI"),
        gt_relative_candidates=(),
        role="train",
    ),
    "fattal": DatasetSpec(
        key="fattal",
        output_name="Fattal",
        input_relative_dir=Path("benchmarks/Fattal"),
        gt_relative_candidates=(),
        role="test",
    ),
    "hsts": DatasetSpec(
        key="hsts",
        output_name="HSTS",
        input_relative_dir=Path("benchmarks/HSTS/input"),
        gt_relative_candidates=(),
        role="test",
    ),
}

# Values printed in Table 1 of CVPR 2025 LHD. Diagnostic only.
PAPER_RTTS_REFERENCE: Dict[str, Dict[str, float]] = {
    "Hazy Input": {
        "FADE": 2.484,
        "Q-Align": 2.0586,
        "CLIPIQA": 0.3882,
        "MUSIQ": 53.768,
        "BRISQUE": 36.6423,
    },
    "DiffDehaze": {
        "FADE": 1.138,
        "Q-Align": 2.8340,
        "CLIPIQA": 0.4263,
        "MUSIQ": 65.086,
        "BRISQUE": 16.4924,
    },
}


@dataclass(frozen=True)
class MetricSpec:
    label: str
    backend: str
    direction: str  # "higher" | "lower"
    kind: str       # "pyiqa" | "fade" | "fr_numpy" | "lpips"


@dataclass(frozen=True)
class EvaluationTarget:
    project_root: Path
    model_name: str
    mode: str
    dataset: str
    dataset_name: str
    dataset_role: str
    source: str
    label: str
    image_root: Path
    gt_root: Optional[Path]
    metrics_root: Path
    # Set only for the new multi-budget HazeCDG output layout.
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
        "PSNR": MetricSpec("PSNR", "rgb_psnr", "higher", "fr_numpy"),
        "SSIM": MetricSpec("SSIM", "skimage_ssim", "higher", "fr_numpy"),
        "LPIPS": MetricSpec("LPIPS", "alex", "lower", "lpips"),
    }


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
    # DFG/scripts/eval_metrics.py -> DFG/
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(str(value)).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def normalize_mode(mode: str) -> str:
    key = str(mode or "").strip().lower()
    if key not in SUPPORTED_MODES:
        raise ValueError(
            "experiment.mode must be one of: " + ", ".join(repr(x) for x in SUPPORTED_MODES)
        )
    return key


def normalize_hazecdg_ablation_mode(mode: str) -> str:
    key = str(mode or "").strip().lower()
    if key not in HAZECDG_ABLATION_MODES:
        raise ValueError(
            "HazeCDG ablation experiment.mode must be one of: "
            + ", ".join(repr(x) for x in HAZECDG_ABLATION_MODES)
        )
    return key


def is_hazecdg_ablation_config(
    cfg: Mapping[str, Any],
    config_path: Optional[Path] = None,
) -> bool:
    """Return True only for the dedicated HazeCDG ablation YAML family.

    Detection deliberately requires both an ablation mode and an ``ablation``
    mapping so legacy LHD/LDRPS/DOD configs cannot be captured accidentally.
    The filename is treated only as an additional hint, never as the sole key.
    """
    mode = str(nested_get(cfg, ["experiment", "mode"], "") or "").strip().lower()
    ablation_section = nested_get(cfg, ["ablation"], None)
    if mode not in HAZECDG_ABLATION_MODES or not isinstance(ablation_section, Mapping):
        return False

    # The current public runner uses the LHD/HazeCDG path keys below. Requiring
    # at least one of them prevents unrelated future configs with a generic
    # ``ablation`` section from being misclassified.
    haze_keys = ("stage1_config", "stage1_checkpoint", "stage2_checkpoint")
    has_haze_path = any(
        str(nested_get(cfg, ["paths", key], "") or "").strip()
        for key in haze_keys
    )
    filename_hint = "hazecdg_ablation" in str(config_path or "").lower()
    return bool(has_haze_path or filename_hint)


def _ablation_beta_label(beta: float) -> str:
    """Mirror eval_HazeCDG_ablation.py::beta_label exactly."""
    value = float(beta)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"Invalid fixed guidance beta={beta}")
    scaled = int(round(value * 100.0))
    if not np.isclose(value, scaled / 100.0, rtol=0.0, atol=1e-10):
        text = (f"{value:.6f}").rstrip("0").rstrip(".").replace(".", "p")
        return f"beta_{text}"
    return f"beta_{scaled:03d}"


def resolve_ablation_variant_names(
    cfg: Mapping[str, Any],
    mode: str,
) -> List[str]:
    """Return image-producing variant directory names in runner order."""
    mode = normalize_hazecdg_ablation_mode(mode)
    if mode == "fixed_strength":
        raw = nested_get(
            cfg, ["ablation", "fixed_strength", "betas"], [0.25, 0.5, 1.0, 2.0]
        )
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
            raise ValueError("ablation.fixed_strength.betas must be a non-empty sequence.")
        names = [_ablation_beta_label(float(beta)) for beta in raw]
        if len(set(names)) != len(names):
            raise ValueError(
                "ablation.fixed_strength.betas map to duplicate output directory names."
            )
        return names + ["adaptive"]
    if mode == "neutral_counter":
        return ["neutral", "counter"]
    if mode == "hold_space":
        return ["epsilon_hold", "score_hold"]
    raise ValueError(
        "experiment.mode='runtime' produces timing CSV/JSON rather than restored "
        "image variants, so image-quality metrics are not applicable."
    )


def detect_model_output_name(
    cfg: Mapping[str, Any],
    config_path: Optional[Path] = None,
) -> str:
    """Detect which model family produced the selected outputs.

    Preferred:
        evaluation.model_name: LHD | LDRPS | DOD | BiLaLoRA | HNDiff

    Legacy external-method YAMLs are still auto-detected from their paths.
    LHD remains the backward-compatible default.
    """

    explicit = str(
        nested_get(cfg, ["evaluation", "model_name"], "") or ""
    ).strip()

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

    hints = [
        str(config_path or ""),
        str(nested_get(cfg, ["paths", "repo_root"], "") or ""),
        str(nested_get(cfg, ["paths", "ld_rps_repo_root"], "") or ""),
        str(nested_get(cfg, ["paths", "model_id"], "") or ""),
        str(nested_get(cfg, ["paths", "checkpoint"], "") or ""),
        str(nested_get(cfg, ["paths", "base_checkpoint"], "") or ""),
        str(nested_get(cfg, ["paths", "lora_checkpoint"], "") or ""),
    ]
    joined = " ".join(hints).lower()

    if "ld-rps" in joined or "ldrps" in joined:
        return LDRPS_OUTPUT_NAME

    if "eval_dod_hazecdg" in joined or "/dod/" in joined or "\\dod\\" in joined:
        return DOD_OUTPUT_NAME

    if "bilalora" in joined:
        return BILALORA_OUTPUT_NAME

    if "hndiff" in joined:
        return HNDIFF_OUTPUT_NAME

    return MODEL_OUTPUT_NAME


def model_uses_mode_layout(model_name: str) -> bool:
    return str(model_name) in MODE_AWARE_MODEL_FAMILIES


def normalize_model_mode(model_name: str, mode: str) -> str:
    key = normalize_mode(mode)
    if model_name == LDRPS_OUTPUT_NAME and key not in {"baseline", "hazecdg"}:
        raise ValueError(
            "LDRPS evaluation supports experiment.mode='baseline' or 'hazecdg'."
        )
    if model_name == DOD_OUTPUT_NAME and key not in {"baseline", "hazecdg"}:
        raise ValueError(
            "DOD evaluation supports experiment.mode='baseline' or 'hazecdg'."
        )
    return key


def resolve_inference_step_count(cfg: Mapping[str, Any], model_name: str) -> int:
    """Return the nominal reverse-evaluation count used to normalize HazeCDG K.

    LHD uses inference.steps. LD-RPS uses the first text-to-image reverse
    trajectory length, inference.text2img_steps. This value is used only for
    reporting the nominal sensing ratio in comparison tables.
    """
    if model_name == LDRPS_OUTPUT_NAME:
        raw = nested_get(
            cfg,
            ["ldrps", "text2img_steps"],
            nested_get(cfg, ["inference", "text2img_steps"], 450),
        )
    elif model_name == DOD_OUTPUT_NAME:
        raw = nested_get(cfg, ["inference", "steps"], 1)
    else:
        raw = nested_get(cfg, ["inference", "steps"], 50)
    steps = int(raw)
    if steps <= 0:
        raise ValueError(f"Invalid inference step count for {model_name}: {steps}.")
    return steps


def normalize_dataset_name(name: str) -> str:
    key = str(name or "").strip().lower()
    if key not in DATASET_ALIASES:
        supported = ", ".join(spec.output_name for spec in DATASETS.values())
        raise ValueError(f"Unsupported data.dataset={name!r}. Supported: {supported}")
    return DATASET_ALIASES[key]


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
        nested_get(cfg, ["data", "dataset_root"], "./datasets/dehaze"),
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


def resolve_dataset_gt_root(
    project_root: Path,
    cfg: Mapping[str, Any],
    dataset: str,
) -> Optional[Path]:
    """Resolve paired GT without guessing for datasets declared unpaired.

    Explicit YAML wins:
        evaluation.gt_root: /path/or/relative/path
        data.gt_root: /path/or/relative/path

    Otherwise only dataset-specific candidates are considered. This avoids
    accidentally treating unrelated LIVE500Foggy folders as clean references.
    """
    explicit = str(
        nested_get(cfg, ["evaluation", "gt_root"], "")
        or nested_get(cfg, ["data", "gt_root"], "")
        or ""
    ).strip()
    if explicit:
        path = resolve_path(project_root, explicit)
        if not path.is_dir():
            raise FileNotFoundError(f"Explicit GT directory does not exist: {path}")
        if not list_images(path):
            raise RuntimeError(f"Explicit GT directory contains no images: {path}")
        return path

    dataset_root = resolve_dataset_root(project_root, cfg)
    for relative in DATASETS[dataset].gt_relative_candidates:
        candidate = (dataset_root / relative).resolve()
        if candidate.is_dir() and list_images(candidate):
            return candidate
    return None


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

    if model_uses_mode_layout(model_name):
        return (
            project_root
            / "outputs"
            / model_name
            / mode
            / DATASETS[dataset].output_name
        ).resolve()

    return (
        project_root
        / "outputs"
        / model_name
        / DATASETS[dataset].output_name
    ).resolve()


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
                f"--mode is only supported for LHD/LDRPS/DOD families, not {model_name}."
            )
        mode = model_name.lower()

    dataset = normalize_dataset_name(nested_get(cfg, ["data", "dataset"], "RTTS"))
    dataset_spec = DATASETS[dataset]
    dataset_name = dataset_spec.output_name
    gt_root = resolve_dataset_gt_root(project_root, cfg, dataset)

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
        if metrics_root_override
        else default_metrics_root.resolve()
    )

    return EvaluationTarget(
        project_root=project_root.resolve(),
        model_name=model_name,
        mode=mode,
        dataset=dataset,
        dataset_name=dataset_name,
        dataset_role=dataset_spec.role,
        source=source,
        label=label,
        image_root=image_root.resolve(),
        gt_root=gt_root.resolve() if gt_root is not None else None,
        metrics_root=metrics_root,
    )




def parse_hazecdg_anchor_counts(cfg: Mapping[str, Any]) -> List[int]:
    """Return deterministic HazeCDG sensing budgets from ``hazecdg.anchor_counts``."""
    raw = nested_get(cfg, ["hazecdg", "anchor_counts"], None)
    if raw is None:
        raise ValueError("hazecdg.anchor_counts is required for multi-K HazeCDG evaluation.")
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
    """Resolve K005/K010/... outputs produced by the multi-budget HazeCDG runner.

    Default layout for mode-aware families:
        outputs/<MODEL>/hazecdg/<DATASET>/K005/
                                         metrics/

    If data.output_root is explicitly configured, Kxxx directories are searched
    underneath that custom root.  If --metrics-root is supplied, it is treated
    as a comparison-metrics base and Kxxx subdirectories are created under it.
    """
    dataset = normalize_dataset_name(nested_get(cfg, ["data", "dataset"], "RTTS"))
    spec = DATASETS[dataset]
    gt_root = resolve_dataset_gt_root(project_root, cfg, dataset)
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
                dataset_name=spec.output_name,
                dataset_role=spec.role,
                source="output",
                label=f"HazeCDG K={k}",
                image_root=image_root,
                gt_root=gt_root.resolve() if gt_root is not None else None,
                metrics_root=metrics_root,
                k=int(k),
            )
        )
    return targets


def resolve_hazecdg_ablation_targets(
    project_root: Path,
    cfg: Mapping[str, Any],
    metrics_root_override: str = "",
    config_path: Optional[Path] = None,
) -> Tuple[List[EvaluationTarget], Path]:
    """Resolve every image-producing variant for one HazeCDG ablation mode.

    Default layout matches ``eval_HazeCDG_ablation.py`` exactly::

        outputs/LHD/ablations/<MODE>/<DATASET>/<VARIANT>/
                                               metrics/

    ``data.output_root`` replaces the ``.../<MODE>/<DATASET>`` base exactly as
    in the inference runner. ``--metrics-root`` affects metric artifacts only;
    image discovery continues to use the inference output root.
    """
    if not is_hazecdg_ablation_config(cfg, config_path=config_path):
        raise ValueError("Config is not a recognized HazeCDG ablation configuration.")

    mode = normalize_hazecdg_ablation_mode(
        nested_get(cfg, ["experiment", "mode"], "fixed_strength")
    )
    variant_names = resolve_ablation_variant_names(cfg, mode)
    dataset = normalize_dataset_name(nested_get(cfg, ["data", "dataset"], "RTTS"))
    spec = DATASETS[dataset]
    gt_root = resolve_dataset_gt_root(project_root, cfg, dataset)

    configured_output = str(nested_get(cfg, ["data", "output_root"], "") or "").strip()
    if configured_output:
        base_output = resolve_path(project_root, configured_output)
    else:
        base_output = (
            project_root
            / "outputs"
            / MODEL_OUTPUT_NAME
            / "ablations"
            / mode
            / spec.output_name
        ).resolve()

    custom_metrics_base = (
        resolve_path(project_root, metrics_root_override)
        if str(metrics_root_override or "").strip()
        else None
    )

    targets: List[EvaluationTarget] = []
    for variant in variant_names:
        image_root = (base_output / variant).resolve()
        metrics_root = (
            (custom_metrics_base / variant).resolve()
            if custom_metrics_base is not None
            else (image_root / "metrics").resolve()
        )
        targets.append(
            EvaluationTarget(
                project_root=project_root.resolve(),
                model_name=MODEL_OUTPUT_NAME,
                mode=mode,
                dataset=dataset,
                dataset_name=spec.output_name,
                dataset_role=spec.role,
                source="output",
                label=variant,
                image_root=image_root,
                gt_root=gt_root.resolve() if gt_root is not None else None,
                metrics_root=metrics_root,
                k=None,
            )
        )
    return targets, base_output.resolve()


def write_hazecdg_comparison(
    base_output_root: Path,
    results: Sequence[Mapping[str, Any]],
    profile: Mapping[str, MetricSpec],
    model_name: str = MODEL_OUTPUT_NAME,
) -> None:
    """Write one K-budget comparison table after all HazeCDG targets finish."""
    if not results:
        return
    base_output_root.mkdir(parents=True, exist_ok=True)

    # Preserve the canonical metric order; include FR metrics only if present.
    metric_labels = [
        m for m in ALL_METRICS
        if any(m in (r.get("summary") or {}) for r in results)
    ]

    rows: List[Dict[str, Any]] = []
    for r in sorted(results, key=lambda x: int(x["K"])):
        row: Dict[str, Any] = {
            "K": int(r["K"]),
            "sensing_ratio": float(r.get("sensing_ratio", float("nan"))),
            "images": int(r.get("images", 0)),
        }
        summary = r.get("summary") or {}
        for m in metric_labels:
            stats = summary.get(m)
            row[m] = "" if not stats else float(stats["mean"])
        rows.append(row)

    csv_path = base_output_root / "comparison.csv"
    fields = ["K", "sensing_ratio", "images"] + metric_labels
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    payload = {
        "mode": "hazecdg",
        "results": rows,
        "metric_directions": {
            m: profile[m].direction for m in metric_labels if m in profile
        },
    }
    (base_output_root / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    width = 16 + 12 * len(metric_labels)
    lines: List[str] = []
    lines.append("=" * max(112, width))
    lines.append(f"{model_name} / HazeCDG — MULTI-BUDGET METRIC COMPARISON")
    lines.append("=" * max(112, width))
    header = f"{'K':>5} {'Sense':>8} {'N':>6}"
    for m in metric_labels:
        arrow = "↑" if profile[m].direction == "higher" else "↓"
        header += f" {m + arrow:>11}"
    lines.append(header)
    lines.append("-" * max(112, width))
    for row in rows:
        sense = row["sensing_ratio"]
        sense_text = "-" if not np.isfinite(sense) else f"{100.0*sense:.1f}%"
        line = f"{int(row['K']):>5d} {sense_text:>8} {int(row['images']):>6d}"
        for m in metric_labels:
            val = row.get(m, "")
            line += f" {('-' if val == '' else f'{float(val):.6f}'):>11}"
        lines.append(line)
    lines.append("=" * max(112, width))
    lines.append("Direction: FADE↓, BRISQUE↓; Q-Align↑, CLIPIQA↑, MUSIQ↑.")
    text = "\n".join(lines) + "\n"
    (base_output_root / "comparison.txt").write_text(text, encoding="utf-8")
    print("\n" + text, end="")
    print(f"Comparison CSV : {csv_path}")


def write_ablation_comparison(
    base_output_root: Path,
    mode: str,
    results: Sequence[Mapping[str, Any]],
    profile: Mapping[str, MetricSpec],
) -> None:
    """Write a compact cross-variant table for one HazeCDG ablation family."""
    if not results:
        return
    mode = normalize_hazecdg_ablation_mode(mode)
    if mode == "runtime":
        raise ValueError("runtime has its own timing.csv/timing.json and no IQA comparison.")

    base_output_root.mkdir(parents=True, exist_ok=True)
    metric_labels = [
        metric
        for metric in ALL_METRICS
        if any(metric in (result.get("summary") or {}) for result in results)
    ]

    rows: List[Dict[str, Any]] = []
    for result in results:
        variant = str(result.get("variant", "")).strip()
        if not variant:
            raise ValueError("Ablation comparison result is missing its variant name.")
        row: Dict[str, Any] = {
            "variant": variant,
            "images": int(result.get("images", 0)),
        }
        summary = result.get("summary") or {}
        for metric in metric_labels:
            stats = summary.get(metric)
            row[metric] = "" if not stats else float(stats["mean"])
        rows.append(row)

    csv_path = base_output_root / "comparison.csv"
    fields = ["variant", "images"] + metric_labels
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "experiment": "HazeCDG ablation",
        "mode": mode,
        "results": rows,
        "metric_directions": {
            metric: profile[metric].direction
            for metric in metric_labels
            if metric in profile
        },
    }
    (base_output_root / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    width = max(112, 28 + 12 * len(metric_labels))
    lines: List[str] = []
    lines.append("=" * width)
    lines.append(f"HazeCDG ABLATION — {mode}")
    lines.append("=" * width)
    header = f"{'Variant':<20} {'N':>6}"
    for metric in metric_labels:
        arrow = "↑" if profile[metric].direction == "higher" else "↓"
        header += f" {metric + arrow:>11}"
    lines.append(header)
    lines.append("-" * width)
    for row in rows:
        line = f"{row['variant']:<20} {int(row['images']):>6d}"
        for metric in metric_labels:
            value = row.get(metric, "")
            line += f" {('-' if value == '' else f'{float(value):.6f}'):>11}"
        lines.append(line)
    lines.append("=" * width)
    lines.append(
        "Direction: FADE↓, BRISQUE↓; Q-Align↑, CLIPIQA↑, MUSIQ↑."
    )
    text = "\n".join(lines) + "\n"
    (base_output_root / "comparison.txt").write_text(text, encoding="utf-8")
    print("\n" + text, end="")
    print(f"Comparison CSV : {csv_path}")


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


def _strip_known_suffixes(stem: str) -> List[str]:
    values = [stem]
    suffixes = ("_hazy", "_haze", "_foggy", "_dehazed", "_restored", "_output")
    lower = stem.lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            values.append(stem[: -len(suffix)])
    return values


def _gt_candidate_stems(output_path: Path, dataset: str) -> List[str]:
    stem = output_path.stem
    candidates: List[str] = []
    for value in _strip_known_suffixes(stem):
        if value not in candidates:
            candidates.append(value)

    # Haze4K commonly uses hazy names such as id_beta_A while GT is id.png.
    if dataset == "haze4k" and "_" in stem:
        prefix = stem.split("_", 1)[0]
        if prefix not in candidates:
            candidates.append(prefix)
    return candidates


def build_gt_pairs(
    image_paths: Sequence[Path],
    image_root: Path,
    gt_root: Path,
    dataset: str,
) -> Dict[Path, Path]:
    """Build strict one-to-one evaluated-image -> GT mapping.

    A GT may legitimately serve multiple hazy images (e.g. synthetic haze
    variants), but each evaluated image must resolve to exactly one GT.
    """
    gt_paths = list_images(gt_root)
    if not gt_paths:
        raise RuntimeError(f"No GT images found under: {gt_root}")

    by_stem: Dict[str, List[Path]] = {}
    by_rel_stem: Dict[str, List[Path]] = {}
    for gt in gt_paths:
        by_stem.setdefault(gt.stem.lower(), []).append(gt)
        rel_stem = gt.relative_to(gt_root).with_suffix("").as_posix().lower()
        by_rel_stem.setdefault(rel_stem, []).append(gt)

    pairs: Dict[Path, Path] = {}
    failures: List[str] = []

    for raw_path in image_paths:
        path = raw_path.resolve()
        rel = path.relative_to(image_root.resolve())
        rel_stem = rel.with_suffix("").as_posix().lower()

        matches: List[Path] = []
        exact_rel = by_rel_stem.get(rel_stem, [])
        if len(exact_rel) == 1:
            matches = exact_rel
        elif len(exact_rel) > 1:
            failures.append(f"ambiguous relative GT for {rel.as_posix()}")
            continue

        if not matches:
            for candidate_stem in _gt_candidate_stems(path, dataset):
                candidate_matches = by_stem.get(candidate_stem.lower(), [])
                if len(candidate_matches) == 1:
                    matches = candidate_matches
                    break
                if len(candidate_matches) > 1:
                    failures.append(
                        f"ambiguous GT stem '{candidate_stem}' for {rel.as_posix()}"
                    )
                    matches = []
                    break

        if len(matches) != 1:
            if not any(rel.as_posix() in f for f in failures):
                failures.append(f"no GT match for {rel.as_posix()}")
            continue
        pairs[path] = matches[0].resolve()

    if failures:
        preview = "\n  - ".join(failures[:10])
        extra = "" if len(failures) <= 10 else f"\n  ... and {len(failures) - 10} more"
        raise RuntimeError(
            f"GT pairing failed for {len(failures)}/{len(image_paths)} evaluated images.\n"
            f"  - {preview}{extra}\n"
            "Full-reference metrics are intentionally strict; no images were silently skipped."
        )
    return pairs


def parse_metric_selection(raw: str) -> List[str]:
    aliases = {
        "fade": "FADE",
        "qalign": "Q-Align",
        "q-align": "Q-Align",
        "clipiqa": "CLIPIQA",
        "clip-iqa": "CLIPIQA",
        "musiq": "MUSIQ",
        "brisque": "BRISQUE",
        "psnr": "PSNR",
        "ssim": "SSIM",
        "lpips": "LPIPS",
    }
    selected: List[str] = []
    for token in raw.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"Unknown metric token: {token!r}")
        label = aliases[key]
        if label not in selected:
            selected.append(label)
    return selected


def choose_metrics(
    metric_profile: str,
    dataset: str,
    gt_available: bool,
    explicit: str = "",
    full_reference_enabled: bool = True,
) -> List[str]:
    """Choose the metric set before looking at method results."""
    explicit = str(explicit or "").strip()
    profile = str(metric_profile or "auto").strip().lower()
    if profile not in {"auto", "primary", "paper"}:
        raise ValueError("metric_profile must be auto | primary | paper")

    if explicit:
        if explicit.lower() == "all":
            selected = list(NR_PAPER_METRICS)
            if gt_available and full_reference_enabled:
                selected += FR_METRICS
            return selected
        selected = parse_metric_selection(explicit)
        requested_fr = [m for m in selected if m in FR_METRICS]
        if requested_fr and not gt_available:
            raise ValueError(
                "Full-reference metric(s) requested but no GT is available: "
                + ", ".join(requested_fr)
            )
        if not full_reference_enabled:
            selected = [m for m in selected if m not in FR_METRICS]
        return selected

    if profile == "paper":
        selected = list(NR_PAPER_METRICS)
    elif profile == "primary":
        selected = list(NR_PRIMARY_METRICS)
    else:  # auto
        # Use the five selected no-reference metrics for every dataset.  This keeps
        # HazeCDG K-budget comparisons on one consistent protocol and also preserves the
        # LHD paper metric set for RTTS.
        selected = list(NR_PAPER_METRICS)

    if gt_available and full_reference_enabled:
        selected += FR_METRICS
    return selected


def read_existing_csv(csv_path: Path, image_keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {k: {} for k in image_keys}
    if not csv_path.is_file():
        return rows
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("image", "")
            if key not in rows:
                continue
            for metric in ALL_METRICS:
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
    columns = ["image"] + ALL_METRICS
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for key in image_keys:
            row: Dict[str, Any] = {"image": key}
            for metric in columns[1:]:
                value = scores.get(key, {}).get(metric)
                row[metric] = "" if value is None else f"{float(value):.10f}"
            writer.writerow(row)
    os.replace(tmp, csv_path)


def metric_is_complete(
    metric_label: str,
    image_keys: Sequence[str],
    scores: Mapping[str, Mapping[str, float]],
) -> bool:
    return bool(image_keys) and all(metric_label in scores.get(k, {}) for k in image_keys)


def _progress(metric: str, idx: int, total: int, path: Path, score: float) -> None:
    if idx == 1 or idx == total or idx % 25 == 0:
        print(f"[{metric:8s}] {idx:04d}/{total:04d} {path.name} -> {score:.6f}")


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
            "PyIQA metrics require IQA-PyTorch. For the paper-time profile, install "
            "pyiqa==0.1.13."
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
        "FADE backend not available. Install official MATLAB FADE or 'fade-python'."
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
    with tempfile.TemporaryDirectory(prefix="dfg_fade_") as td:
        td_path = Path(td)
        image_list = td_path / "images.txt"
        score_file = td_path / "scores.txt"
        script_file = td_path / "run_fade.m"

        image_list.write_text(
            "\n".join(str(p.resolve()) for p in image_paths) + "\n", encoding="utf-8"
        )
        script = f"""
addpath(genpath('{_matlab_quote(fade_dir)}'));
fid_in = fopen('{_matlab_quote(image_list)}', 'r');
fid_out = fopen('{_matlab_quote(score_file)}', 'w');
if fid_in < 0 || fid_out < 0
    error('Could not open LHD FADE temporary files.');
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
        cmd = [matlab_cmd, "-batch", f"run('{_matlab_quote(script_file)}')"]
        subprocess.run(cmd, check=True)

        lines = score_file.read_text(encoding="utf-8").splitlines()
        if len(lines) != len(image_paths):
            raise RuntimeError(
                f"MATLAB FADE returned {len(lines)} scores for {len(image_paths)} images."
            )
        for idx, (line, path, key) in enumerate(zip(lines, image_paths, image_keys), 1):
            fields = line.split(",")
            if len(fields) != 2:
                raise RuntimeError(f"Malformed MATLAB FADE output: {line}")
            score = float(fields[1])
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
        raise RuntimeError("Could not import PyFADE. Install package 'fade-python'.") from exc

    print("\n[metric] FADE <- PyFADE (unofficial MATLAB-aligned implementation)")
    for idx, (path, key) in enumerate(zip(image_paths, image_keys), 1):
        score = float(fade(str(path)))
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite FADE score for {path}: {score}")
        scores[key]["FADE"] = score
        _progress("FADE", idx, len(image_paths), path, score)
        if idx % 25 == 0:
            write_per_image_csv(csv_path, image_keys, scores)
    write_per_image_csv(csv_path, image_keys, scores)


def _load_rgb_float(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _prepare_fr_arrays(
    output_path: Path,
    gt_path: Path,
    crop_border: int,
    resize_output_to_gt: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    out = _load_rgb_float(output_path)
    gt = _load_rgb_float(gt_path)

    if out.shape != gt.shape:
        if not resize_output_to_gt:
            raise RuntimeError(
                "Output/GT size mismatch for full-reference evaluation:\n"
                f"  output: {output_path} -> {out.shape}\n"
                f"  gt    : {gt_path} -> {gt.shape}\n"
                "Do not silently resize for paper evaluation. If resizing is intentional, "
                "use --resize-output-to-gt."
            )
        h, w = gt.shape[:2]
        with Image.open(output_path) as img:
            img = img.convert("RGB")
            try:
                resample = Image.Resampling.BICUBIC
            except AttributeError:
                resample = Image.BICUBIC
            img = img.resize((w, h), resample=resample)
            out = np.asarray(img, dtype=np.float32) / 255.0

    if crop_border < 0:
        raise ValueError("crop_border must be >= 0")
    if crop_border > 0:
        h, w = out.shape[:2]
        if 2 * crop_border >= min(h, w):
            raise ValueError(
                f"crop_border={crop_border} is too large for image size {w}x{h}."
            )
        out = out[crop_border:-crop_border, crop_border:-crop_border, :]
        gt = gt[crop_border:-crop_border, crop_border:-crop_border, :]
    return out, gt


def _psnr_rgb(out: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((out.astype(np.float64) - gt.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def _ssim_rgb(out: np.ndarray, gt: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except Exception as exc:
        raise RuntimeError(
            "SSIM requires scikit-image. Install it with: pip install scikit-image"
        ) from exc
    return float(structural_similarity(gt, out, data_range=1.0, channel_axis=2))


def evaluate_fr_numpy_metric(
    label: str,
    image_paths: Sequence[Path],
    image_keys: Sequence[str],
    gt_pairs: Mapping[Path, Path],
    scores: Dict[str, Dict[str, float]],
    csv_path: Path,
    crop_border: int,
    resize_output_to_gt: bool,
) -> None:
    print(f"\n[metric] {label} <- paired RGB full-reference")
    for idx, (path, key) in enumerate(zip(image_paths, image_keys), 1):
        gt = gt_pairs[path.resolve()]
        out_arr, gt_arr = _prepare_fr_arrays(
            path, gt, crop_border=crop_border, resize_output_to_gt=resize_output_to_gt
        )
        score = _psnr_rgb(out_arr, gt_arr) if label == "PSNR" else _ssim_rgb(out_arr, gt_arr)
        if not math.isfinite(score) and label != "PSNR":
            raise RuntimeError(f"Non-finite {label} score for {path}: {score}")
        scores[key][label] = score
        _progress(label, idx, len(image_paths), path, score)
        if idx % 25 == 0:
            write_per_image_csv(csv_path, image_keys, scores)
    write_per_image_csv(csv_path, image_keys, scores)


def evaluate_lpips_metric(
    image_paths: Sequence[Path],
    image_keys: Sequence[str],
    gt_pairs: Mapping[Path, Path],
    scores: Dict[str, Dict[str, float]],
    csv_path: Path,
    device: str,
    crop_border: int,
    resize_output_to_gt: bool,
) -> None:
    try:
        import torch
        import lpips
    except Exception as exc:
        raise RuntimeError(
            "LPIPS requires the 'lpips' package. Install it with: pip install lpips"
        ) from exc

    print(f"\n[metric] LPIPS <- official lpips package, net='alex' on {device}")
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()

    with torch.inference_mode():
        for idx, (path, key) in enumerate(zip(image_paths, image_keys), 1):
            gt = gt_pairs[path.resolve()]
            out_arr, gt_arr = _prepare_fr_arrays(
                path, gt, crop_border=crop_border, resize_output_to_gt=resize_output_to_gt
            )
            out_t = (
                torch.from_numpy(out_arr.copy())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )
            gt_t = (
                torch.from_numpy(gt_arr.copy())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )
            out_t = out_t * 2.0 - 1.0
            gt_t = gt_t * 2.0 - 1.0
            score = scalarize(model(out_t, gt_t))
            if not math.isfinite(score):
                raise RuntimeError(f"Non-finite LPIPS score for {path}: {score}")
            scores[key]["LPIPS"] = score
            _progress("LPIPS", idx, len(image_paths), path, score)
            if idx % 25 == 0:
                write_per_image_csv(csv_path, image_keys, scores)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_per_image_csv(csv_path, image_keys, scores)


def compute_summary(
    image_keys: Sequence[str],
    scores: Mapping[str, Mapping[str, float]],
    metric_labels: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for metric in metric_labels:
        vals = [float(scores[k][metric]) for k in image_keys if metric in scores.get(k, {})]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        result[metric] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": int(arr.size),
        }
    return result


def paper_reference_for(target: EvaluationTarget) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
    if target.model_name != MODEL_OUTPUT_NAME:
        return None, None
    if target.dataset != "rtts":
        return None, None
    if target.source == "input":
        return "Hazy Input", PAPER_RTTS_REFERENCE["Hazy Input"]
    if target.mode == "baseline":
        return "DiffDehaze", PAPER_RTTS_REFERENCE["DiffDehaze"]
    return None, None


def write_summary_files(
    target: EvaluationTarget,
    profile: Mapping[str, MetricSpec],
    selected_metrics: Sequence[str],
    metric_profile_name: str,
    image_count: int,
    gt_pair_count: int,
    summary: Mapping[str, Mapping[str, float]],
    fade_backend: Optional[str],
    pyiqa_version: Optional[str],
    lpips_version: Optional[str],
    crop_border: int,
    resize_output_to_gt: bool,
    elapsed_seconds: float,
) -> None:
    target.metrics_root.mkdir(parents=True, exist_ok=True)
    ref_name, ref = paper_reference_for(target)

    payload: Dict[str, Any] = {
        "model": target.model_name,
        "source": target.source,
        "mode": target.mode,
        "dataset": target.dataset_name,
        "dataset_role": target.dataset_role,
        "image_root": str(target.image_root),
        "gt_root": str(target.gt_root) if target.gt_root else None,
        "image_count": image_count,
        "gt_pair_count": gt_pair_count,
        "metric_profile": metric_profile_name,
        "selected_metrics": list(selected_metrics),
        "metrics": summary,
        "metric_backends": {label: profile[label].backend for label in selected_metrics},
        "fade_backend": fade_backend,
        "pyiqa_version": pyiqa_version,
        "recommended_pyiqa_version": RECOMMENDED_PYIQA_VERSION,
        "lpips_version": lpips_version,
        "full_reference_protocol": {
            "rgb_range": "[0,1]",
            "crop_border": crop_border,
            "resize_output_to_gt": resize_output_to_gt,
            "ssim": "skimage.metrics.structural_similarity(channel_axis=2, data_range=1.0)",
            "lpips_net": "alex",
        },
        "elapsed_seconds": elapsed_seconds,
    }
    if ref_name and ref:
        payload["paper_reference"] = {
            "name": ref_name,
            "values": ref,
            "note": (
                "The DiffDehaze Table-1 reference uses AccSamp (50 steps, tau=800, "
                "omega=600, s=0.1)."
                if ref_name == "DiffDehaze"
                else "Paper Table-1 RTTS hazy-input sanity-check row."
            ),
        }
        payload["paper_delta"] = {
            metric: float(summary[metric]["mean"] - ref[metric])
            for metric in ref
            if metric in summary
        }

    json_path = target.metrics_root / "summary.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines: List[str] = []
    lines.append("=" * 112)
    lines.append(f"{target.model_name} — unified metric summary")
    lines.append("=" * 112)
    lines.append(f"Model        : {target.model_name}")
    lines.append(f"Source       : {target.source}")
    lines.append(f"Mode         : {target.mode}")
    lines.append(f"Dataset      : {target.dataset_name} ({target.dataset_role})")
    lines.append(f"Image root   : {target.image_root}")
    lines.append(f"GT root      : {target.gt_root if target.gt_root else 'NONE'}")
    lines.append(f"Images       : {image_count}")
    lines.append(f"GT pairs     : {gt_pair_count}")
    lines.append(f"Profile      : {metric_profile_name}")
    lines.append(f"PyIQA        : {pyiqa_version or 'not used'}")
    lines.append(f"FADE backend : {fade_backend or 'not used'}")
    lines.append(f"LPIPS pkg    : {lpips_version or 'not used'}")
    if gt_pair_count:
        lines.append(f"FR protocol  : RGB, crop={crop_border}, resize_output_to_gt={resize_output_to_gt}")
    lines.append("-" * 112)
    lines.append(
        f"{'Metric':<12} {'Direction':<10} {'Mean':>14} {'Std':>14} "
        f"{'Count':>8} {'Paper ref':>14} {'Delta':>14}"
    )
    lines.append("-" * 112)

    for label in selected_metrics:
        if label not in summary:
            continue
        spec = profile[label]
        mean = summary[label]["mean"]
        std = summary[label]["std"]
        count = int(summary[label]["count"])
        if ref and label in ref:
            ref_text = f"{ref[label]:.6f}"
            delta_text = f"{mean - ref[label]:+.6f}"
        else:
            ref_text = "-"
            delta_text = "-"
        arrow = "↑" if spec.direction == "higher" else "↓"
        lines.append(
            f"{label:<12} {(arrow + ' ' + spec.direction):<10} {mean:>14.6f} {std:>14.6f} "
            f"{count:>8d} {ref_text:>14} {delta_text:>14}"
        )

    lines.append("=" * 112)
    if target.dataset == "rtts" and image_count != EXPECTED_RTTS_COUNT:
        lines.append(
            f"WARNING: RTTS paper evaluation uses {EXPECTED_RTTS_COUNT} images; this run used {image_count}."
        )
    if target.dataset_role == "train":
        lines.append(
            "WARNING: URHI is HazeGen training data. Do NOT use this result as a paper test benchmark."
        )
    if ref_name == "DiffDehaze":
        lines.append("NOTE: Paper DiffDehaze reference is the AccSamp result.")
    if pyiqa_version and pyiqa_version != RECOMMENDED_PYIQA_VERSION:
        lines.append(
            f"WARNING: installed pyiqa={pyiqa_version}; paper-time profile recommends "
            f"pyiqa=={RECOMMENDED_PYIQA_VERSION}."
        )
    if fade_backend == "pyfade":
        lines.append("NOTE: PyFADE is unofficial; MATLAB FADE is preferred for paper reproduction.")
    if resize_output_to_gt:
        lines.append(
            "WARNING: outputs were resized to GT for FR metrics. Report this protocol explicitly."
        )

    text = "\n".join(lines) + "\n"
    (target.metrics_root / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text, end="")
    print(f"CSV          : {target.metrics_root / 'per_image_metrics.csv'}")
    print(f"JSON         : {json_path}")


def evaluate_one_target(
    *,
    project_root: Path,
    config_path: Path,
    cfg: Mapping[str, Any],
    target: EvaluationTarget,
    args,
) -> Dict[str, Any]:
    """Evaluate one resolved image directory and return its summary metadata."""
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

    image_keys = [p.relative_to(target.image_root).as_posix() for p in image_paths]
    if len(set(image_keys)) != len(image_keys):
        raise RuntimeError("Duplicate relative image paths detected.")

    yaml_profile = str(nested_get(cfg, ["evaluation", "metric_profile"], "") or "").strip()
    metric_profile_name = args.metric_profile
    if args.metric_profile == "auto" and yaml_profile in {"primary", "paper"}:
        metric_profile_name = yaml_profile

    full_reference_enabled = not args.no_full_reference
    yaml_fr_enabled = nested_get(cfg, ["evaluation", "full_reference", "enabled"], None)
    if yaml_fr_enabled is False:
        full_reference_enabled = False

    selected = choose_metrics(
        metric_profile=metric_profile_name,
        dataset=target.dataset,
        gt_available=target.gt_root is not None,
        explicit=args.metrics,
        full_reference_enabled=full_reference_enabled,
    )

    profile = build_metric_profile(args.brisque_variant)
    device = args.device.strip() or str(nested_get(cfg, ["runtime", "device"], "cuda:0"))

    crop_border = args.crop_border
    if crop_border < 0:
        crop_border = int(nested_get(cfg, ["evaluation", "full_reference", "crop_border"], 0) or 0)
    if crop_border < 0:
        raise ValueError("crop_border must be >= 0")

    resize_output_to_gt = bool(args.resize_output_to_gt)
    if not resize_output_to_gt:
        resize_output_to_gt = bool(
            nested_get(cfg, ["evaluation", "full_reference", "resize_output_to_gt"], False)
        )

    gt_pairs: Dict[Path, Path] = {}
    selected_fr = [m for m in selected if m in FR_METRICS]
    if selected_fr:
        if target.gt_root is None:
            raise RuntimeError("Internal error: FR metrics selected without GT root.")
        gt_pairs = build_gt_pairs(
            image_paths=image_paths,
            image_root=target.image_root,
            gt_root=target.gt_root,
            dataset=target.dataset,
        )

    target.metrics_root.mkdir(parents=True, exist_ok=True)
    csv_path = target.metrics_root / "per_image_metrics.csv"
    scores = read_existing_csv(csv_path, image_keys)
    if args.force:
        for key in image_keys:
            for metric in selected:
                scores[key].pop(metric, None)

    print("=" * 112)
    print("Unified dehazing metric evaluation")
    print("=" * 112)
    print(f"Config       : {config_path}")
    print(f"Model        : {target.model_name}")
    print(f"Source       : {target.source}")
    print(f"Mode         : {target.mode}")
    if target.k is not None:
        print(f"HazeCDG K       : {target.k}")
    print(f"Dataset      : {target.dataset_name}")
    print(f"Dataset role : {target.dataset_role}")
    print(f"Image root   : {target.image_root}")
    print(f"GT root      : {target.gt_root if target.gt_root else 'NONE'}")
    print(f"Images       : {len(image_paths)}")
    print(f"GT pairs     : {len(gt_pairs)}")
    print(f"Profile      : {metric_profile_name}")
    print(f"Metrics      : {', '.join(selected)}")
    print(f"Metric output: {target.metrics_root}")
    print(f"Device       : {device}")
    if selected_fr:
        print(f"FR crop      : {crop_border}")
        print(f"FR resize    : {resize_output_to_gt}")
    print("=" * 112)

    if target.dataset_role == "train":
        print(
            "[WARNING] URHI is HazeGen training data. Diagnostic use only; "
            "do not report it as an independent paper test benchmark."
        )

    pyiqa_version = safe_package_version("pyiqa")
    if any(profile[m].kind == "pyiqa" for m in selected):
        if pyiqa_version is None:
            raise RuntimeError(
                "pyiqa is not installed. For closest paper-time reproduction use: "
                f"pyiqa=={RECOMMENDED_PYIQA_VERSION}"
            )
        print(f"[env] pyiqa version : {pyiqa_version}")
        if pyiqa_version != RECOMMENDED_PYIQA_VERSION:
            print(
                f"[warn] Recommended paper-time profile is pyiqa=={RECOMMENDED_PYIQA_VERSION}; "
                f"installed={pyiqa_version}."
            )

    lpips_version = safe_package_version("lpips") if "LPIPS" in selected else None
    fade_backend_used: Optional[str] = None
    start = time.time()

    for label in selected:
        spec = profile[label]
        if not args.force and metric_is_complete(label, image_keys, scores):
            print(f"[skip] {label}: complete values already exist in {csv_path}")
            continue

        if spec.kind == "fade":
            fade_dir = resolve_path(project_root, args.fade_matlab_dir)
            fade_backend_used = choose_fade_backend(args.fade_backend, fade_dir, args.matlab_cmd)
            if fade_backend_used == "matlab":
                evaluate_fade_matlab(
                    image_paths, image_keys, scores, fade_dir, args.matlab_cmd, csv_path
                )
            else:
                evaluate_fade_pyfade(image_paths, image_keys, scores, csv_path)
        elif spec.kind == "pyiqa":
            evaluate_pyiqa_metric(spec, image_paths, image_keys, scores, device, csv_path)
        elif spec.kind == "fr_numpy":
            evaluate_fr_numpy_metric(
                label=label,
                image_paths=image_paths,
                image_keys=image_keys,
                gt_pairs=gt_pairs,
                scores=scores,
                csv_path=csv_path,
                crop_border=crop_border,
                resize_output_to_gt=resize_output_to_gt,
            )
        elif spec.kind == "lpips":
            evaluate_lpips_metric(
                image_paths=image_paths,
                image_keys=image_keys,
                gt_pairs=gt_pairs,
                scores=scores,
                csv_path=csv_path,
                device=device,
                crop_border=crop_border,
                resize_output_to_gt=resize_output_to_gt,
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
        metric_profile_name=metric_profile_name,
        image_count=len(image_paths),
        gt_pair_count=len(gt_pairs),
        summary=summary,
        fade_backend=fade_backend_used,
        pyiqa_version=pyiqa_version,
        lpips_version=lpips_version,
        crop_border=crop_border,
        resize_output_to_gt=resize_output_to_gt,
        elapsed_seconds=elapsed,
    )
    return {
        "target": target,
        "summary": summary,
        "selected_metrics": selected,
        "profile": profile,
        "images": len(image_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default="./configs/eval_LHD_HazeCDG.yaml")
    parser.add_argument(
        "--source",
        choices=["output", "input"],
        default="output",
        help="Evaluate model outputs, or original dataset inputs.",
    )
    parser.add_argument(
        "--mode",
        choices=list(SUPPORTED_MODES),
        default="",
        help="Optional LHD/LDRPS/DOD mode override. Ignored for other external-method YAMLs.",
    )
    parser.add_argument(
        "--metric-profile",
        choices=["auto", "primary", "paper"],
        default="auto",
        help="Metric bundle. --metrics overrides this.",
    )
    parser.add_argument(
        "--metrics",
        default="",
        help="Comma-separated exact override, or 'all'. Empty uses --metric-profile.",
    )
    parser.add_argument(
        "--no-full-reference",
        action="store_true",
        help="Do not auto-add PSNR/SSIM/LPIPS even when paired GT exists.",
    )
    parser.add_argument("--device", default="", help="Metric device; empty uses runtime.device from YAML.")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images found.")
    parser.add_argument("--force", action="store_true", help="Recompute selected metrics already in CSV.")
    parser.add_argument(
        "--metrics-root",
        default="",
        help=(
            "Optional custom metric output directory. For multi-K HazeCDG this is "
            "treated as a base and K005/K010/... metric directories are created under it."
        ),
    )
    parser.add_argument(
        "--brisque-variant", choices=["brisque_matlab", "brisque"], default="brisque_matlab"
    )
    parser.add_argument(
        "--fade-backend", choices=["auto", "matlab", "pyfade"], default="auto"
    )
    parser.add_argument("--fade-matlab-dir", default="./third_party/FADE")
    parser.add_argument("--matlab-cmd", default="matlab")
    parser.add_argument(
        "--crop-border",
        type=int,
        default=-1,
        help="FR border crop. -1 uses evaluation.full_reference.crop_border or 0.",
    )
    parser.add_argument(
        "--resize-output-to-gt",
        action="store_true",
        help="Explicitly resize output to GT size for FR metrics. Default is strict size equality.",
    )
    args = parser.parse_args()

    if args.max_images < 0:
        raise ValueError("--max-images must be >= 0.")

    project_root = project_root_from_script()
    config_path = resolve_path(project_root, args.config)
    cfg = load_yaml(config_path)

    is_ablation_cfg = is_hazecdg_ablation_config(cfg, config_path=config_path)
    if is_ablation_cfg and args.source == "output":
        if args.mode:
            raise ValueError(
                "For eval_HazeCDG_ablation.yaml, select the experiment with "
                "experiment.mode in YAML rather than --mode."
            )

        ablation_mode = normalize_hazecdg_ablation_mode(
            nested_get(cfg, ["experiment", "mode"], "fixed_strength")
        )
        if ablation_mode == "runtime":
            runtime_dataset = normalize_dataset_name(
                nested_get(cfg, ["data", "dataset"], "RTTS")
            )
            configured_output = str(
                nested_get(cfg, ["data", "output_root"], "") or ""
            ).strip()
            runtime_root = (
                resolve_path(project_root, configured_output)
                if configured_output
                else (
                    project_root
                    / "outputs"
                    / MODEL_OUTPUT_NAME
                    / "ablations"
                    / "runtime"
                    / DATASETS[runtime_dataset].output_name
                ).resolve()
            )
            print("=" * 112)
            print("HazeCDG ablation metric evaluation")
            print("=" * 112)
            print("Mode         : runtime")
            print(f"Runtime root : {runtime_root}")
            print(
                "No image-quality evaluation is run for runtime mode. "
                "Use timing.csv / timing_per_image.csv / timing.json produced by "
                "eval_HazeCDG_ablation.py."
            )
            print("=" * 112)
            return

        targets, base_output = resolve_hazecdg_ablation_targets(
            project_root,
            cfg,
            metrics_root_override=args.metrics_root,
            config_path=config_path,
        )
        print("=" * 112)
        print("HazeCDG ablation metric evaluation")
        print("=" * 112)
        print(f"Mode         : {ablation_mode}")
        print(f"Dataset      : {targets[0].dataset_name}")
        print(f"Variants     : {[target.label for target in targets]}")
        print(f"Base output  : {base_output}")
        print("Metrics      : FADE, Q-Align, CLIPIQA, MUSIQ, BRISQUE")
        print("=" * 112)

        ablation_results: List[Dict[str, Any]] = []
        for target in targets:
            print("\n" + "#" * 112)
            print(f"# HazeCDG ablation | {ablation_mode} | variant={target.label}")
            print("#" * 112)
            out = evaluate_one_target(
                project_root=project_root,
                config_path=config_path,
                cfg=cfg,
                target=target,
                args=args,
            )
            ablation_results.append(
                {
                    "variant": target.label,
                    "images": int(out["images"]),
                    "summary": out["summary"],
                }
            )

        profile = build_metric_profile(args.brisque_variant)
        write_ablation_comparison(
            base_output, ablation_mode, ablation_results, profile
        )
        return

    # ``--source input`` remains supported even with the ablation YAML. The
    # ablation mode is irrelevant when evaluating the original dataset images,
    # so use a shallow normalized copy solely for the legacy input resolver.
    cfg_for_standard = cfg
    if is_ablation_cfg and args.source == "input":
        cfg_for_standard = dict(cfg)
        cfg_for_standard["experiment"] = dict(cfg.get("experiment", {}))
        cfg_for_standard["experiment"]["mode"] = "baseline"

    model_name = detect_model_output_name(cfg_for_standard, config_path=config_path)
    if model_uses_mode_layout(model_name):
        requested_mode = normalize_model_mode(
            model_name,
            args.mode or nested_get(cfg_for_standard, ["experiment", "mode"], "baseline"),
        )
    else:
        if args.mode:
            raise ValueError(
                f"--mode is only supported for LHD/LDRPS/DOD families, not {model_name}."
            )
        requested_mode = model_name.lower()

    is_new_multik_hazecdg = (
        model_name in MULTIK_HAZECDG_MODEL_FAMILIES
        and args.source == "output"
        and requested_mode == "hazecdg"
        and nested_get(cfg_for_standard, ["hazecdg", "anchor_counts"], None) is not None
    )

    if is_new_multik_hazecdg:
        targets = resolve_hazecdg_multik_targets(
            project_root,
            cfg_for_standard,
            metrics_root_override=args.metrics_root,
            model_name=model_name,
        )
        base_output = resolve_output_root(
            project_root, cfg_for_standard, "hazecdg", targets[0].dataset, model_name=model_name
        )
        reference_steps = resolve_inference_step_count(cfg_for_standard, model_name)
        print("=" * 112)
        print(f"{model_name} HazeCDG multi-budget evaluation")
        print("=" * 112)
        print(f"Dataset      : {targets[0].dataset_name}")
        print(f"K values     : {[t.k for t in targets]}")
        print(f"Base output  : {base_output}")
        print(f"Ref. steps   : {reference_steps}")
        print("Metrics      : FADE, Q-Align, CLIPIQA, MUSIQ, BRISQUE")
        print("=" * 112)

        results: List[Dict[str, Any]] = []
        for target in targets:
            assert target.k is not None
            sensing_ratio = float(target.k) / float(reference_steps)
            print("\n" + "#" * 112)
            print(
                f"# HazeCDG K={target.k}  "
                f"({100.0 * sensing_ratio:.1f}% of {reference_steps} reverse evaluations)"
            )
            print("#" * 112)
            out = evaluate_one_target(
                project_root=project_root,
                config_path=config_path,
                cfg=cfg_for_standard,
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

    # Single-output LHD/LDRPS/DOD modes and external methods share the same metric implementation.
    target = resolve_evaluation_target(
        project_root,
        cfg_for_standard,
        source=args.source,
        metrics_root_override=args.metrics_root,
        mode_override=args.mode,
        config_path=config_path,
    )
    evaluate_one_target(
        project_root=project_root,
        config_path=config_path,
        cfg=cfg_for_standard,
        target=target,
        args=args,
    )


if __name__ == "__main__":
    main()


