#!/usr/bin/env python3
"""
DOD (Diffusion Once and Done, AAAI 2026) runner with HazeCDG guidance.

Intended project location
-------------------------
    DFG/scripts/eval_DOD_HazeCDG.py

Controlled modes
----------------
    baseline
        Official DOD one-step restoration using the released Stage1/Stage2
        checkpoints, SD2.1 backbone, MFM degradation encoder, and Stage2 HDE
        decoder. LHD HazeGen is not loaded.

    hazecdg
        The exact same DOD one-step inference with one HazeCDG correction
        inserted immediately before DOD's scheduler step. At DOD's native
        latent state and timestep t=999, the frozen LHD Stage1 branch provides

            g = eps_D - eps_base,

        where eps_D is the haze-conditioned epsilon prediction and eps_base is
        the corresponding frozen-backbone prediction without the HazeGen
        ControlNet residual. The counter-degradation correction is

            q      = 2 g,
            eps_R' = eps_R - 2 g.

        The factor 2 follows from symmetric transport across the backbone
        reference: eps_D = eps_base + g and its counter-degradation counterpart
        is eps_base - g, so the displacement between them is exactly 2g.
        DOD then performs its original one-step scheduler update and original
        Stage2 HDE decoding. No model weights are updated.

One-step setting
----------------
DOD is an official one-step restoration diffusion model. There is exactly one
restoration UNet evaluation at native timestep t=999, so HazeCDG performs one
sensing/correction call. Sparse anchor scheduling is not applicable.

Haze signal
-----------
For hazecdg mode only, the released LHD Stage1 HazeGen is loaded as a frozen
sensing model. At DOD's exact lq_latent and exact t=999 state:

    eps_D    = Stage1 SD2.1 UNet with Stage1 ControlNet residuals
    eps_base = the same frozen Stage1 SD2.1 UNet without ControlNet residuals

The validated zero latent hint is retained. The DOD latent itself is not
re-encoded by LHD; it is passed directly to the LHD Stage1 epsilon field.

Compatibility checks
--------------------
Before HazeCDG inference the runner verifies:
  - DOD scheduler prediction_type == "epsilon"
  - DOD uses 1000 training timesteps
  - DOD restoration timestep is exactly 999
  - DOD and LHD Stage1 beta schedules match numerically
  - DOD and LHD latent scale factors match
  - both restoration/sensing UNets operate on 4-channel latents

This prevents a silent epsilon/v/sample parameterization mismatch.

Preprocessing
-------------
Both baseline and hazecdg follow DOD's official image-space contract:
  1) if min(H,W) < process_size (official default 256), upscale the short side
     to process_size while preserving aspect ratio,
  2) apply optional integer upscale (official default 1),
  3) resize H and W down to multiples of 8 using Lanczos,
  4) map RGB [0,1] to [-1,1],
  5) optionally resize the final output back to the original HxW.

HazeCDG does not change the DOD image or DOD latent resolution. Only the frozen
LHD Stage1 sensing branch receives a temporary symmetric replicate-pad of the
DOD latent to the next multiple of 8 in latent space. eps_D and eps_base are
then cropped back to the exact original DOD latent HxW before guidance. Thus
baseline and hazecdg use the same DOD input/state and preprocessing.

Output folders
--------------
    baseline : outputs/DOD/baseline/<DATASET>/
    hazecdg  : outputs/DOD/hazecdg/<DATASET>/
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_F
import yaml


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_OUTPUT_NAME = "DOD"
NUMERICAL_EPS = 1e-12
SUPPORTED_MODES = ("baseline", "hazecdg")
DOD_NATIVE_TIMESTEP = 999
DOD_IMAGE_MULTIPLE = 8
LHD_SENSOR_LATENT_MULTIPLE = 8

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
}

DATASET_OUTPUT_NAMES = {
    "rtts": "RTTS",
    "haze4k": "Haze4K",
    "live500foggy": "LIVE500Foggy",
    "urhi": "URHI",
    "fattal": "Fattal",
}

DATASET_RELATIVE_DIRS = {
    "rtts": Path("test/RTTS"),
    "haze4k": Path("benchmarks/Haze4K/test/haze"),
    "live500foggy": Path("benchmarks/LIVE500Foggy"),
    "urhi": Path("train/hazegen/URHI"),
    "fattal": Path("benchmarks/Fattal"),
}


# =============================================================================
# Generic helpers
# =============================================================================


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")
    return cfg


def nested_get(d: Dict[str, Any], keys: Sequence[str], default=None):
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def project_root_from_script() -> Path:
    # DFG/scripts/eval_DOD_HazeCDG.py -> DFG/
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(str(value)).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def normalize_mode(mode: str) -> str:
    key = str(mode or "").strip().lower()
    if key not in SUPPORTED_MODES:
        raise ValueError(
            "experiment.mode must be one of: "
            + ", ".join(repr(x) for x in SUPPORTED_MODES)
        )
    return key


def normalize_dataset_name(name: str) -> str:
    key = str(name or "").strip().lower()
    if key not in DATASET_ALIASES:
        supported = ", ".join(DATASET_OUTPUT_NAMES.values())
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


def resolve_dataset_input_root(
    dataset_root: Path,
    dataset: str,
    explicit_input_root: str,
    project_root: Path,
) -> Path:
    explicit = str(explicit_input_root or "").strip()
    if explicit:
        path = resolve_path(project_root, explicit)
        if not path.is_dir():
            raise FileNotFoundError(f"Configured data.input_root does not exist: {path}")
        return path

    candidate = (dataset_root / DATASET_RELATIVE_DIRS[dataset]).resolve()
    if candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        f"Could not locate dataset {DATASET_OUTPUT_NAMES[dataset]} automatically.\n"
        f"Expected: {candidate}\n"
        "Set data.input_root explicitly if your dataset is elsewhere."
    )


def default_output_root(project_root: Path, mode: str, dataset: str) -> Path:
    mode = normalize_mode(mode)
    dataset = normalize_dataset_name(dataset)
    return (
        project_root
        / "outputs"
        / MODEL_OUTPUT_NAME
        / mode
        / DATASET_OUTPUT_NAMES[dataset]
    )


def seed_all(seed: int) -> None:
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_compat(path: Path, *, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_state_dict(obj: Any) -> Any:
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def validate_inference_steps(steps: int) -> int:
    steps = int(steps)
    if steps != 1:
        raise ValueError(
            "DOD pretrained inference is an official one-step model. "
            f"Set inference.steps: 1, got {steps}."
        )
    return steps


@contextlib.contextmanager
def working_directory(path: Path) -> Iterator[None]:
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def floor_to_multiple(value: int, multiple: int) -> int:
    value = int(value)
    multiple = int(multiple)
    if value <= 0 or multiple <= 0:
        raise ValueError("value and multiple must be positive.")
    return value - value % multiple


def pad_latent_for_lhd_sensor(
    x: torch.Tensor,
    *,
    multiple: int = LHD_SENSOR_LATENT_MULTIPLE,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Pad only the LHD sensing copy; the DOD latent itself stays untouched.

    Padding is symmetric (difference at most one pixel) and uses replicate
    values to avoid introducing an artificial zero-valued boundary in latent
    space.  The returned crop tuple is (top, bottom, left, right).
    """
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW latent, got shape={tuple(x.shape)}.")
    multiple = int(multiple)
    if multiple <= 0:
        raise ValueError("multiple must be positive.")

    h, w = int(x.shape[-2]), int(x.shape[-1])
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)

    padded = torch_F.pad(x, (left, right, top, bottom), mode="replicate")
    return padded, (top, bottom, left, right)


def crop_sensor_output(
    x: torch.Tensor,
    crop: Tuple[int, int, int, int],
    *,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    top, bottom, left, right = (int(v) for v in crop)
    del bottom, right  # target_hw defines the exact retained extent.
    target_h, target_w = (int(v) for v in target_hw)
    out = x[..., top : top + target_h, left : left + target_w]
    if out.shape[-2:] != (target_h, target_w):
        raise RuntimeError(
            "Failed to crop LHD sensor output back to DOD latent size: "
            f"got={tuple(out.shape[-2:])}, expected={(target_h, target_w)}."
        )
    return out


def tile_starts(length: int, tile_size: int, tile_overlap: int) -> List[int]:
    """Return deterministic starts whose tiles cover [0, length) without gaps."""
    length = int(length)
    tile_size = min(int(tile_size), length)
    tile_overlap = int(tile_overlap)
    if length <= 0 or tile_size <= 0:
        raise ValueError("length and tile_size must be positive.")
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError(
            "tile_overlap must satisfy 0 <= overlap < effective tile_size. "
            f"Got length={length}, tile={tile_size}, overlap={tile_overlap}."
        )
    if length <= tile_size:
        return [0]

    stride = tile_size - tile_overlap
    last = length - tile_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_stitch_accumulators(
    latent: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use FP32 accumulation so tiny Gaussian edge weights do not underflow."""
    shape = tuple(latent.shape)
    device = latent.device
    return (
        torch.zeros(shape, device=device, dtype=torch.float32),
        torch.zeros(shape, device=device, dtype=torch.float32),
    )


# =============================================================================
# Exact counter-degradation guidance core (same rule as current LHD runner)
# =============================================================================


def _guidance_common_diagnostics(
    eps_r: torch.Tensor,
    degradation_residual: torch.Tensor,
    correction: torch.Tensor,
    *,
    eps: float = NUMERICAL_EPS,
) -> Dict[str, float]:
    r = eps_r.detach().float().reshape(-1)
    g = degradation_residual.detach().float().reshape(-1)
    q = correction.detach().float().reshape(-1)

    r_norm_t = torch.linalg.vector_norm(r)
    g_norm_t = torch.linalg.vector_norm(g)
    q_norm_t = torch.linalg.vector_norm(q)

    denom = r_norm_t * g_norm_t
    if bool(torch.isfinite(denom)) and float(denom.item()) > eps:
        cosine = float((torch.dot(r, g) / denom).clamp(-1.0, 1.0).item())
    else:
        cosine = 0.0

    return {
        "eps_r_norm": float(r_norm_t.item()),
        "g_norm": float(g_norm_t.item()),
        "correction_norm": float(q_norm_t.item()),
        "eps_r_g_cosine": cosine,
    }


def apply_counter_degradation_guidance(
    eps_r: torch.Tensor,
    degradation_residual: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Apply the final HazeCDG counter-degradation transport: eps_R' = eps_R - 2g.

    The local haze differential is

        g = eps_D - eps_base.

    Relative to eps_base, eps_D lies at +g and its symmetric
    counter-degradation counterpart lies at -g. The exact displacement between
    those two points is therefore 2g, yielding the parameter-free correction

        q = 2g.

    DOD then applies its one-step scheduler normalization to q before the
    scheduler update.
    """
    if eps_r.shape != degradation_residual.shape:
        raise ValueError(
            "Counter-degradation guidance shape mismatch: "
            f"eps_r={tuple(eps_r.shape)}, "
            f"g={tuple(degradation_residual.shape)}"
        )

    g = degradation_residual.detach().float()

    # Final HazeCDG transport.
    hazecdg_correction = 2.0 * g

    # DOD one-step scheduler normalization used by this integration.
    scheduler_scale = 0.44341345

    correction_f = scheduler_scale * hazecdg_correction
    guided = (eps_r.float() - correction_f).to(dtype=eps_r.dtype)

    diag = _guidance_common_diagnostics(
        eps_r,
        degradation_residual,
        correction_f,
    )
    diag.update(
        {
            "active": 1.0 if diag["g_norm"] > NUMERICAL_EPS else 0.0,
        }
    )
    return guided, diag


# =============================================================================
# Environment / imports
# =============================================================================


def check_conda_environment(expected: str = "LHD_xformers_blackwell") -> None:
    active = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
    if not active:
        raise RuntimeError(
            f"No active conda environment detected. Activate it first:\n  conda activate {expected}"
        )
    if active != expected and Path(active).name != expected:
        raise RuntimeError(f"Wrong conda environment. expected={expected}, active={active}")


def check_torch_cuda(device: str) -> None:
    print(f"[env] Python executable : {sys.executable}")
    print(f"[env] PyTorch           : {torch.__version__}")
    print(f"[env] Torch CUDA        : {torch.version.cuda}")
    print(f"[env] CUDA available    : {torch.cuda.is_available()}")
    if not str(device).startswith("cuda"):
        raise RuntimeError("Official DOD code requires CUDA; use runtime.device: cuda:0.")
    if not torch.cuda.is_available():
        raise RuntimeError(f"runtime.device={device} but CUDA is not available.")
    dev = torch.device(device)
    idx = 0 if dev.index is None else int(dev.index)
    if idx != 0:
        raise RuntimeError(
            "DOD upstream code contains explicit .cuda() calls. Use runtime.device: cuda:0 "
            "and CUDA_VISIBLE_DEVICES to select a physical GPU if needed."
        )
    print(f"[env] GPU               : {torch.cuda.get_device_name(idx)}")


def configure_lhd_attention_backend(mode: str) -> None:
    key = str(mode or "xformers").strip().lower()
    if key not in {"sdp", "xformers", "vanilla"}:
        raise ValueError(
            "runtime.lhd_attention_mode must be one of: sdp | xformers | vanilla"
        )
    os.environ["ATTN_MODE"] = key
    print(f"[env] LHD attention     : {key}")


def verify_xformers_runtime(device: str) -> None:
    try:
        import xformers
        import xformers.ops as xops
    except Exception as exc:
        raise RuntimeError("xformers could not be imported.") from exc

    q = torch.randn(1, 64, 5, 64, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    y = xops.memory_efficient_attention(q, k, v)
    torch.cuda.synchronize()
    print(f"[env] xformers          : {xformers.__version__} (preflight PASS)")
    del q, k, v, y
    torch.cuda.empty_cache()


def import_dod(repo_root: Path):
    repo_root = repo_root.resolve()
    required = [
        repo_root / "osediff.py",
        repo_root / "model_qkv.py",
        repo_root / "guided_diffusion" / "script_util.py",
        repo_root / "models" / "autoencoder_kl.py",
        repo_root / "models" / "unet_2d_condition.py",
        repo_root / "prompt_embeds.pt",
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "DOD checkout is incomplete:\n" + "\n".join(f"  - {p}" for p in missing)
        )

    repo_str = str(repo_root)
    sys.path[:] = [p for p in sys.path if p != repo_str]
    sys.path.insert(0, repo_str)

    # DOD keeps custom SD modules in `DOD/models/` but that directory has no
    # __init__.py.  It is therefore a namespace package named simply `models`.
    # When this runner is launched from DFG/scripts, any unrelated regular
    # package also named `models` can take precedence and make
    # `from models.autoencoder_kl ...` fail even though DOD/models exists.
    # Bind the namespace explicitly while importing osediff, then restore any
    # pre-existing `models` modules so the rest of the process is not polluted.
    previous_models = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "models" or name.startswith("models.")
    }
    for name in previous_models:
        sys.modules.pop(name, None)

    dod_models_pkg = ModuleType("models")
    dod_models_pkg.__path__ = [str((repo_root / "models").resolve())]
    dod_models_pkg.__package__ = "models"
    sys.modules["models"] = dod_models_pkg

    try:
        with working_directory(repo_root):
            from osediff import OSEDiff_test
            from guided_diffusion.script_util import i_DDPM
    except Exception as exc:
        raise RuntimeError(f"Failed to import official DOD runtime from {repo_root}.") from exc
    finally:
        for name in list(sys.modules):
            if name == "models" or name.startswith("models."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_models)

    print(f"[env] DOD repo           : {repo_root}")
    print("[env] DOD imports        : OK")
    return OSEDiff_test, i_DDPM


def import_lhd_sensor_runtime(repo_root: Path):
    repo_root = repo_root.resolve()
    required = [
        repo_root / "diffbir" / "model" / "cldm.py",
        repo_root / "configs" / "inference" / "stage1.yaml",
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Learning-Hazing-to-Dehazing checkout is incomplete:\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    repo_str = str(repo_root)
    sys.path[:] = [p for p in sys.path if p != repo_str]
    sys.path.insert(0, repo_str)
    try:
        from omegaconf import OmegaConf
        from diffbir.utils.common import instantiate_from_config
    except Exception as exc:
        raise RuntimeError(f"Failed to import LHD sensing runtime from {repo_root}.") from exc

    print(f"[env] LHD repo           : {repo_root}")
    print("[env] LHD sensor imports : OK")
    return OmegaConf, instantiate_from_config


# =============================================================================
# LHD Stage1 sensing model
# =============================================================================


class LHDStage1Sensor:
    """Frozen Stage1 SD2.1 UNet + HazeGen ControlNet + text encoder only."""

    def __init__(
        self,
        *,
        unet,
        controlnet,
        clip,
        text_condition: torch.Tensor,
        scale_factor: float,
        betas: np.ndarray,
        device: torch.device,
    ) -> None:
        self.unet = unet
        self.controlnet = controlnet
        self.clip = clip
        self.text_condition = text_condition.detach()
        self.scale_factor = float(scale_factor)
        self.betas = np.asarray(betas, dtype=np.float64)
        self.device = device

    @torch.no_grad()
    def predict_degradation_and_base(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[0] != 1:
            raise RuntimeError(
                "DOD+HazeCDG currently supports batch-size-1 latent sensing only; "
                f"got shape={tuple(x.shape)}."
            )

        # Preserve the validated LHD Stage1 FP32 sensing path.  DOD's fp16
        # latent is only cast for the frozen sensor; no re-encoding occurs.
        # Crucially, padding is applied ONLY to this sensing copy.
        orig_h, orig_w = int(x.shape[-2]), int(x.shape[-1])
        x_sensor = x.detach().to(device=self.device, dtype=torch.float32)
        x_sensor, crop = pad_latent_for_lhd_sensor(
            x_sensor, multiple=LHD_SENSOR_LATENT_MULTIPLE
        )

        t_sensor = timesteps.detach().to(device=self.device, dtype=torch.long)
        context = self.text_condition.to(device=self.device)
        if context.shape[0] != x_sensor.shape[0]:
            context = context.expand(x_sensor.shape[0], *context.shape[1:])

        zero_hint = torch.zeros_like(x_sensor)
        control = self.controlnet(
            x=x_sensor,
            hint=zero_hint,
            timesteps=t_sensor,
            context=context,
        )
        control = [c * 1.0 for c in control]

        # ControlledUnetModel mutates the control list via pop(), hence copy.
        eps_deg = self.unet(
            x=x_sensor,
            timesteps=t_sensor,
            context=context,
            control=list(control),
            only_mid_control=False,
        )
        eps_base = self.unet(
            x=x_sensor,
            timesteps=t_sensor,
            context=context,
            control=None,
            only_mid_control=False,
        )

        # Remove sensor-only context padding before constructing g_t.
        eps_deg = crop_sensor_output(
            eps_deg, crop, target_hw=(orig_h, orig_w)
        )
        eps_base = crop_sensor_output(
            eps_base, crop, target_hw=(orig_h, orig_w)
        )
        if eps_deg.shape != x.shape or eps_base.shape != x.shape:
            raise RuntimeError(
                "LHD sensor output shape does not match the untouched DOD latent: "
                f"latent={tuple(x.shape)}, eps_D={tuple(eps_deg.shape)}, "
                f"eps_base={tuple(eps_base.shape)}."
            )
        return eps_deg, eps_base


def load_lhd_stage1_sensor(
    *,
    config_path: Path,
    sd_checkpoint: Path,
    controlnet_checkpoint: Path,
    degradation_prompt: str,
    device: torch.device,
    OmegaConf,
    instantiate_from_config,
) -> LHDStage1Sensor:
    cfg = OmegaConf.load(str(config_path))
    model = instantiate_from_config(cfg.model.cldm)

    sd_blob = torch_load_compat(sd_checkpoint, map_location="cpu")
    sd_state = unwrap_state_dict(sd_blob)
    if not isinstance(sd_state, dict):
        raise RuntimeError(f"Unexpected SD checkpoint format: {sd_checkpoint}")
    unused, missing = model.load_pretrained_sd(sd_state)
    print(f"[env] G_D/stage1 SD          : {sd_checkpoint}")
    print(f"[env] G_D/stage1 SD missing  : {len(missing)}")
    print(f"[env] G_D/stage1 SD unused   : {len(unused)}")

    control_blob = torch_load_compat(controlnet_checkpoint, map_location="cpu")
    control_state = unwrap_state_dict(control_blob)
    if not isinstance(control_state, dict):
        raise RuntimeError(f"Unexpected ControlNet checkpoint: {controlnet_checkpoint}")
    model.load_controlnet_from_ckpt(control_state)
    print(f"[env] G_D/stage1 ControlNet  : {controlnet_checkpoint}")

    diffusion = instantiate_from_config(cfg.model.diffusion)
    if str(diffusion.parameterization) != "eps":
        raise RuntimeError(
            "LHD Stage1 must use epsilon parameterization for HazeCDG, "
            f"got {diffusion.parameterization!r}."
        )
    stage1_betas = np.asarray(diffusion.betas, dtype=np.float64).copy()
    scale_factor = float(model.scale_factor)

    # DOD supplies the latent.  LHD VAE is therefore unnecessary at inference.
    unet = model.unet
    unet.train(False)
    unet.requires_grad_(False)
    unet.to(device)

    controlnet = model.controlnet
    controlnet.train(False)
    controlnet.requires_grad_(False)
    controlnet.to(device)
    clip = model.clip
    clip.train(False)
    clip.requires_grad_(False)
    clip.to(device)
    with torch.no_grad():
        text_condition = clip.encode([degradation_prompt]).detach()

    # Detach retained modules so deleting the container releases the unused VAE.
    model.unet = None
    model.controlnet = None
    model.clip = None
    del diffusion, model, sd_blob, sd_state, control_blob, control_state
    gc.collect()
    torch.cuda.empty_cache()

    print("[HazeCDG-DOD] LHD Stage1 VRAM policy: UNet + ControlNet + CLIP only")
    print("[HazeCDG-DOD] LHD Stage1 VAE        : not resident / DOD latent used directly")

    return LHDStage1Sensor(
        unet=unet,
        controlnet=controlnet,
        clip=clip,
        text_condition=text_condition,
        scale_factor=scale_factor,
        betas=stage1_betas,
        device=device,
    )


# =============================================================================
# Official DOD model / MFM loading
# =============================================================================


def build_dod_args(cfg: "RuntimeConfig") -> SimpleNamespace:
    return SimpleNamespace(
        pretrained_model_name_or_path=str(cfg.dod_sd21_root),
        osediff_path=str(cfg.dod_stage1_checkpoint),
        nafnet_path=str(cfg.dod_stage2_checkpoint),
        mixed_precision=cfg.mixed_precision,
        latent_tiled_size=cfg.latent_tiled_size,
        latent_tiled_overlap=cfg.latent_tiled_overlap,
        lora_rank=cfg.lora_rank,
        # Kept for upstream compatibility even if OSEDiff_test does not use all.
        vae_decoder_tiled_size=cfg.vae_decoder_tiled_size,
        vae_encoder_tiled_size=cfg.vae_encoder_tiled_size,
        merge_and_unload_lora=False,
    )


def load_official_dod_model(OSEDiff_test, cfg: "RuntimeConfig"):
    args = build_dod_args(cfg)
    # Upstream uses relative imports during construction in some versions.
    with working_directory(cfg.dod_repo_root):
        model = OSEDiff_test(args)
    model.eval()
    model.requires_grad_(False)
    print(f"[env] DOD Stage1         : {cfg.dod_stage1_checkpoint}")
    print(f"[env] DOD Stage2         : {cfg.dod_stage2_checkpoint}")
    print(f"[env] DOD SD2.1          : {cfg.dod_sd21_root}")
    return model


def convert_norm_module_to_float(module: nn.Module) -> None:
    if isinstance(module, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)):
        module.float()
    for child in module.children():
        convert_norm_module_to_float(child)


def load_official_dod_mfm(i_DDPM, checkpoint: Path, device: torch.device):
    ddpm_model, _ = i_DDPM()
    blob = torch_load_compat(checkpoint, map_location="cpu")
    if not isinstance(blob, dict):
        raise RuntimeError(f"Unexpected DOD MFM checkpoint format: {checkpoint}")
    filtered_state_dict = {
        k: v
        for k, v in blob.items()
        if "output_blocks" not in k and ("out.0" not in k) and ("out.2" not in k)
    }
    ddpm_model.load_state_dict(filtered_state_dict, strict=True)
    ddpm_model = ddpm_model.to(device).half()
    convert_norm_module_to_float(ddpm_model)
    ddpm_model.train(False)
    ddpm_model.requires_grad_(False)

    context_processor = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(start_dim=1),
    ).to(device)
    context_processor.train(False)
    context_processor.requires_grad_(False)

    print(f"[env] DOD MFM DDPM       : {checkpoint}")
    return ddpm_model, context_processor


# =============================================================================
# Compatibility guardrails
# =============================================================================


def validate_dod_lhd_compatibility(dod_model, sensor: LHDStage1Sensor) -> None:
    prediction_type = str(getattr(dod_model.sched.config, "prediction_type", ""))
    if prediction_type != "epsilon":
        raise RuntimeError(
            "DOD scheduler is not epsilon-prediction compatible: "
            f"prediction_type={prediction_type!r}."
        )

    n_train = int(getattr(dod_model.sched.config, "num_train_timesteps", -1))
    if n_train != 1000:
        raise RuntimeError(f"DOD expected 1000 train timesteps, got {n_train}.")

    ts = dod_model.timesteps.detach().cpu().reshape(-1).tolist()
    if ts != [DOD_NATIVE_TIMESTEP]:
        raise RuntimeError(
            f"DOD pretrained restoration timestep must be [{DOD_NATIVE_TIMESTEP}], got {ts}."
        )

    dod_scale = float(dod_model.vae.config.scaling_factor)
    if not np.isclose(dod_scale, sensor.scale_factor, rtol=1e-8, atol=1e-10):
        raise RuntimeError(
            "DOD and LHD Stage1 latent scales differ: "
            f"DOD={dod_scale}, LHD={sensor.scale_factor}."
        )

    dod_betas = dod_model.sched.betas.detach().cpu().double().numpy()
    lhd_betas = np.asarray(sensor.betas, dtype=np.float64)

    max_diff = float("inf")
    if dod_betas.shape == lhd_betas.shape:
        max_diff = float(np.max(np.abs(dod_betas - lhd_betas)))

    # DOD/diffusers stores the SD2.1 schedule through torch float32, while
    # the LHD schedule is constructed in NumPy precision.  The two schedules
    # are considered identical up to harmless floating-point quantization.
    if dod_betas.shape != lhd_betas.shape or not np.allclose(
            dod_betas,
            lhd_betas,
            rtol=1e-6,
            atol=1e-8,
    ):
        raise RuntimeError(
            "DOD SD2.1 and LHD Stage1 beta schedules differ. "
            f"shapes={dod_betas.shape}/{lhd_betas.shape}, "
            f"max_abs_diff={max_diff:.3e}."
        )

    print(
        "[compat] DOD/LHD beta schedule : PASS "
        f"| max_abs_diff={max_diff:.3e}"
    )

    dod_channels = int(getattr(dod_model.unet.config, "in_channels", -1))
    lhd_channels = int(getattr(sensor.unet, "in_channels", -1))
    if dod_channels != 4 or lhd_channels != 4:
        raise RuntimeError(
            f"Expected 4-channel latent UNets, got DOD={dod_channels}, LHD={lhd_channels}."
        )

    print("[compat] DOD prediction      : epsilon")
    print("[compat] DOD native timestep : 999")
    print("[compat] Train timesteps     : 1000")
    print(f"[compat] Latent scale        : {dod_scale:.8f} (MATCH)")
    print(
        "[compat] Beta schedule       : MATCH "
        f"(max|Δ|={np.max(np.abs(dod_betas - sensor.betas)):.3e})"
    )
    print("[compat] Latent channels     : 4 (MATCH)")
    print("[compat] eps_R/eps_D space   : epsilon parameterization compatible")
    print(
        "[compat] NOTE                : DOD uses its trained one-step E(y),t=999 "
        "state; this experiment tests whether HazeGen's differential remains useful there."
    )


# =============================================================================
# Exact DOD one-step forward with one insertion point
# =============================================================================


def _set_dod_conditional_lora(model, degra_context: torch.Tensor) -> None:
    unet_de_c_embed = model.unet_de_mlp(degra_context)
    unet_block_c_embeds = model.unet_block_mlp(model.unet_block_embeddings.weight)
    unet_embeds = model.unet_fuse_mlp(
        torch.cat(
            [
                unet_de_c_embed.unsqueeze(1).repeat(
                    1, unet_block_c_embeds.shape[0], 1
                ),
                unet_block_c_embeds.unsqueeze(0).repeat(
                    unet_de_c_embed.shape[0], 1, 1
                ),
            ],
            -1,
        )
    )

    for layer_name, module in model.unet.named_modules():
        if layer_name not in model.unet_lora_layers:
            continue
        split_name = layer_name.split(".")
        if split_name[0] == "down_blocks":
            block_id = int(split_name[1])
            if block_id >= unet_embeds.shape[1]:
                raise RuntimeError(
                    f"DOD LoRA block_id={block_id} exceeds condition embeddings."
                )
            unet_embed = unet_embeds[:, block_id]
        elif split_name[0] == "mid_block":
            unet_embed = unet_embeds[:, 4]
        elif split_name[0] == "up_blocks":
            unet_embed = unet_embeds[:, int(split_name[1]) + 5]
        else:
            unet_embed = unet_embeds[:, -1]
        module.gamma, module.beta = torch.chunk(unet_embed, chunks=2, dim=1)


def _dod_predict_epsilon(
    model,
    lq_latent: torch.Tensor,
    prompt_embeds: torch.Tensor,
    c_t: torch.Tensor,
) -> torch.Tensor:
    """Reproduce DOD direct/tiled epsilon prediction with robust FP32 stitching.

    The released DOD Gaussian mask contains very small edge weights.  When the
    stitching accumulator inherits an fp16 latent dtype, those values can
    underflow to zero and falsely leave uncovered pixels.  We therefore keep
    tile inference in the model dtype but accumulate weighted predictions and
    contributors in FP32, then cast the stitched epsilon back to the original
    latent dtype.
    """
    _, _, h, w = lq_latent.size()
    configured_tile = int(model.latent_tiled_size)
    configured_overlap = int(model.latent_tiled_overlap)

    if h * w <= configured_tile * configured_tile:
        print("[DOD] Tiled latent: unnecessary for this input")
        return model.unet(
            lq_latent,
            model.timesteps,
            encoder_hidden_states=prompt_embeds,
        ).sample

    tile_size = min(configured_tile, h, w)
    tile_overlap = min(configured_overlap, tile_size - 1)
    x_starts = tile_starts(w, tile_size, tile_overlap)
    y_starts = tile_starts(h, tile_size, tile_overlap)

    print(
        f"[DOD] Tiled latent: input={c_t.shape[-1]}x{c_t.shape[-2]} "
        f"latent={w}x{h} tile={tile_size} overlap={tile_overlap} "
        f"grid={len(x_starts)}x{len(y_starts)}"
    )

    tile_weights = model._gaussian_weights(tile_size, tile_size, 1).to(
        device=lq_latent.device, dtype=torch.float32
    )
    noise_pred, contributors = make_stitch_accumulators(lq_latent)

    for y in y_starts:
        for x in x_starts:
            input_tile = lq_latent[
                :, :, y : y + tile_size, x : x + tile_size
            ]
            model_out = model.unet(
                input_tile,
                model.timesteps,
                encoder_hidden_states=prompt_embeds,
            ).sample

            region = (
                slice(None),
                slice(None),
                slice(y, y + tile_size),
                slice(x, x + tile_size),
            )
            noise_pred[region] += model_out.float() * tile_weights
            contributors[region] += tile_weights

    min_contributor = float(contributors.min().item())
    if (not np.isfinite(min_contributor)) or min_contributor <= 0.0:
        raise RuntimeError(
            "DOD tiled epsilon stitching produced invalid coverage: "
            f"min_contributor={min_contributor:.4e}, latent={w}x{h}, "
            f"tile={tile_size}, overlap={tile_overlap}."
        )

    stitched = noise_pred / contributors
    if not bool(torch.isfinite(stitched).all()):
        raise RuntimeError("DOD tiled epsilon stitching produced non-finite values.")
    return stitched.to(dtype=lq_latent.dtype)


@torch.no_grad()
def dod_one_step_forward(
    model,
    c_t: torch.Tensor,
    degra_context: torch.Tensor,
    prompt_embeds: torch.Tensor,
    *,
    sensor: Optional[LHDStage1Sensor] = None,
) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
    """Run official DOD inference with HazeCDG inserted only before sched.step."""
    c_t = c_t.to(model.device, dtype=model.weight_dtype)
    degra_context = degra_context.to(model.device, dtype=model.weight_dtype)
    prompt_embeds = prompt_embeds.to(model.device, dtype=model.weight_dtype)

    # Official DOD conditional-LoRA preparation.
    _set_dod_conditional_lora(model, degra_context)

    # Official DOD Stage1 VAE encoder, including released encoder LoRA.
    lq_latent = (
        model.vae.encode(c_t).latent_dist.sample() * model.vae.config.scaling_factor
    )
    skip_feats = model.vae.encoder.current_down_blocks

    # Official DOD one-step SD2.1 epsilon prediction (direct or tiled).
    eps_r = _dod_predict_epsilon(model, lq_latent, prompt_embeds, c_t)
    diag: Optional[Dict[str, float]] = None

    # ======================================================================
    # THE ONLY HazeCDG INSERTION POINT
    # official: eps_r -> sched.step(...)
    # HazeCDG : eps_r -> eps_r - 2*(eps_D - eps_base) -> sched.step(...)
    # ======================================================================
    if sensor is not None:
        eps_deg, eps_base = sensor.predict_degradation_and_base(
            lq_latent, model.timesteps
        )
        eps_deg = eps_deg.to(device=eps_r.device, dtype=torch.float32)
        eps_base = eps_base.to(device=eps_r.device, dtype=torch.float32)
        g_t = eps_deg - eps_base
        eps_for_step, diag = apply_counter_degradation_guidance(
            eps_r, g_t
        )
        del eps_deg, eps_base, g_t
    else:
        eps_for_step = eps_r

    # Official DOD one-step scheduler update: unchanged except epsilon argument.
    x_denoised = model.sched.step(
        eps_for_step,
        model.timesteps,
        lq_latent,
        return_dict=True,
    ).prev_sample

    # Official released Stage2 HDE decoder: unchanged.
    x_denoised = model.vae.post_quant_conv(
        x_denoised.to(model.device, dtype=model.weight_dtype)
        / model.vae.config.scaling_factor
    )
    output_image = model.vae_decoder.decoder(x_denoised, skip_feats).clamp(-1, 1)

    del lq_latent, eps_r, eps_for_step, x_denoised, skip_feats
    return output_image, diag


# =============================================================================
# Runtime config
# =============================================================================


@dataclass
class RuntimeConfig:
    project_root: Path
    dod_repo_root: Path
    dod_sd21_root: Path
    dod_stage1_checkpoint: Path
    dod_stage2_checkpoint: Path
    dod_mfm_checkpoint: Path
    dod_prompt_embeds: Path
    lhd_repo_root: Path
    lhd_stage1_config: Path
    lhd_sd_checkpoint: Path
    lhd_stage1_checkpoint: Path
    dataset: str
    dataset_root: Path
    input_root: Path
    output_root: Path
    mode: str
    device: str
    expected_conda_env: str
    lhd_attention_mode: str
    cuda_visible_devices: str
    seed: int
    max_images: int
    skip_existing: bool
    steps: int
    process_size: int
    upscale: int
    mixed_precision: str
    lora_rank: int
    latent_tiled_size: int
    latent_tiled_overlap: int
    vae_decoder_tiled_size: int
    vae_encoder_tiled_size: int
    degradation_prompt: str
    save_original_size: bool
    log_diagnostics: bool
    raw_cfg: Dict[str, Any]


class DODRunner:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        if cfg.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices

        check_conda_environment(cfg.expected_conda_env)
        check_torch_cuda(cfg.device)
        self.device = torch.device(cfg.device)

        self._import_image_runtime()
        self.OSEDiff_test, self.i_DDPM = import_dod(cfg.dod_repo_root)
        self.dod_model = load_official_dod_model(self.OSEDiff_test, cfg)
        self.mfm_model, self.context_processor = load_official_dod_mfm(
            self.i_DDPM, cfg.dod_mfm_checkpoint, self.device
        )
        self.prompt_embeds = self._load_prompt_embeds(cfg.dod_prompt_embeds)

        self.sensor: Optional[LHDStage1Sensor] = None
        if cfg.mode == "hazecdg":
            configure_lhd_attention_backend(cfg.lhd_attention_mode)
            if cfg.lhd_attention_mode == "xformers":
                verify_xformers_runtime(cfg.device)
            OmegaConf, instantiate_from_config = import_lhd_sensor_runtime(
                cfg.lhd_repo_root
            )
            self.sensor = load_lhd_stage1_sensor(
                config_path=cfg.lhd_stage1_config,
                sd_checkpoint=cfg.lhd_sd_checkpoint,
                controlnet_checkpoint=cfg.lhd_stage1_checkpoint,
                degradation_prompt=cfg.degradation_prompt,
                device=self.device,
                OmegaConf=OmegaConf,
                instantiate_from_config=instantiate_from_config,
            )
            validate_dod_lhd_compatibility(self.dod_model, self.sensor)
        else:
            print("[HazeCDG-DOD] baseline mode: LHD Stage1 HazeGen is NOT loaded.")

    def _import_image_runtime(self) -> None:
        try:
            from PIL import Image
            import torchvision
            from torchvision import transforms
            import torchvision.transforms.functional as TVF
        except Exception as exc:
            raise RuntimeError("DOD inference requires PIL and torchvision.") from exc
        self.Image = Image
        self.torchvision = torchvision
        self.transforms = transforms
        self.TVF = TVF
        self.to_tensor = transforms.ToTensor()

    def _load_prompt_embeds(self, path: Path) -> torch.Tensor:
        prompt = torch_load_compat(path, map_location=self.device)
        if not torch.is_tensor(prompt):
            raise RuntimeError(f"Expected tensor in DOD prompt embeddings: {path}")
        if prompt.ndim != 3:
            raise RuntimeError(
                f"Unexpected DOD prompt embedding shape: {tuple(prompt.shape)}"
            )
        print(f"[env] DOD prompt embeds  : {path} shape={tuple(prompt.shape)}")
        return prompt

    @torch.no_grad()
    def _extract_mfm_context(self, lq_minus1_1: torch.Tensor) -> torch.Tensor:
        t0 = torch.tensor([0], device=lq_minus1_1.device, dtype=torch.long)

        # MFM is the released 256x256 ImageNet diffusion encoder.
        # It provides only a global degradation context, so use its native
        # training resolution independently of the full-resolution DOD branch.
        mfm_input = torch_F.interpolate(
            lq_minus1_1,
            size=(256, 256),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        degra_context = self.mfm_model(mfm_input, t0)
        degra_context = self.context_processor(degra_context.float())

        del mfm_input
        return degra_context

    def _read_and_prepare(self, image_path: Path):
        image = self.Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        work = image
        resize_reason: List[str] = []

        if min(orig_w, orig_h) < self.cfg.process_size:
            scale = float(self.cfg.process_size) / float(min(orig_w, orig_h))
            work = work.resize(
                (max(1, int(scale * orig_w)), max(1, int(scale * orig_h))),
                self.Image.Resampling.LANCZOS,
            )
            resize_reason.append(f"short-side->{self.cfg.process_size}")

        if self.cfg.upscale != 1:
            work = work.resize(
                (work.width * self.cfg.upscale, work.height * self.cfg.upscale),
                self.Image.Resampling.LANCZOS,
            )
            resize_reason.append(f"upscale={self.cfg.upscale}")

        # Preserve DOD's official image-space contract for BOTH baseline and
        # hazecdg.  LHD compatibility is handled later by padding only a sensing
        # copy of the latent; the actual DOD image/latent is never changed.
        multiple = DOD_IMAGE_MULTIPLE
        new_w = floor_to_multiple(work.width, multiple)
        new_h = floor_to_multiple(work.height, multiple)
        if new_w <= 0 or new_h <= 0:
            raise RuntimeError(
                "DOD preprocessing produced non-positive size "
                f"from {work.width}x{work.height} with multiple={multiple}."
            )
        if (new_w, new_h) != work.size:
            work = work.resize((new_w, new_h), self.Image.Resampling.LANCZOS)
            resize_reason.append(f"multiple-of-{multiple} (official DOD)")

        if work.width % multiple != 0 or work.height % multiple != 0:
            raise RuntimeError(
                "DOD preprocessing failed: "
                f"got {work.width}x{work.height}, multiple={multiple}."
            )

        if work.size != (orig_w, orig_h):
            print(
                f"[resize] {orig_w}x{orig_h} -> {work.width}x{work.height} "
                f"({' + '.join(resize_reason)})"
            )

        lq_01 = self.to_tensor(work).unsqueeze(0).to(self.device)
        lq_minus1_1 = self.TVF.normalize(
            lq_01,
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        )
        return lq_minus1_1, orig_h, orig_w, work.height, work.width

    @torch.no_grad()
    def _run_one(self, image_path: Path, image_seed: int):
        seed_all(image_seed)
        lq, orig_h, orig_w, work_h, work_w = self._read_and_prepare(image_path)
        degra_context = self._extract_mfm_context(lq)
        prompt_embeds = self.prompt_embeds.expand(lq.shape[0], -1, -1)

        output_minus1_1, diag = dod_one_step_forward(
            self.dod_model,
            lq,
            degra_context,
            prompt_embeds,
            sensor=self.sensor if self.cfg.mode == "hazecdg" else None,
        )

        result = ((output_minus1_1.float() + 1.0) / 2.0).clamp(0.0, 1.0)
        result = result.detach().cpu()

        if self.cfg.save_original_size and (work_h, work_w) != (orig_h, orig_w):
            result = torch_F.interpolate(
                result,
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp(0.0, 1.0)

        if diag is not None and self.cfg.log_diagnostics:
            print(
                "[HazeCDG-DOD] one-step"
                f" | t={DOD_NATIVE_TIMESTEP}"
                f" | active={int(diag['active'])}"
                f" | ||eps_R||={diag['eps_r_norm']:.4e}"
                f" | ||g||={diag['g_norm']:.4e}"
                f" | ||2g||={diag['correction_norm']:.4e}"
                f" | cos(eps_R,g)={diag['eps_r_g_cosine']:+.4f}"
                f" | corr/eps_R={diag['correction_norm'] / max(diag['eps_r_norm'], NUMERICAL_EPS):.4f}"
            )

        del lq, degra_context, prompt_embeds, output_minus1_1
        return result, {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "work_h": work_h,
            "work_w": work_w,
            "diag": diag,
        }

    def _output_path(self, image_path: Path) -> Path:
        try:
            rel = image_path.relative_to(self.cfg.input_root)
        except ValueError:
            rel = Path(image_path.name)
        return (self.cfg.output_root / rel).with_suffix(".png")

    def _save_tensor_atomic(self, result: torch.Tensor, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(out_path.stem + ".tmp.png")
        self.torchvision.utils.save_image(result.squeeze(0), str(tmp_path))
        os.replace(tmp_path, out_path)
        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise IOError(f"Output verification failed: {out_path}")

    def run(self) -> None:
        files = list_images(self.cfg.input_root)
        if not files:
            raise RuntimeError(f"No input images found under: {self.cfg.input_root}")
        if self.cfg.max_images > 0:
            files = files[: self.cfg.max_images]

        self.cfg.output_root.mkdir(parents=True, exist_ok=True)
        print("=" * 108)
        print("DOD :: official one-step restoration + exact counter-degradation guidance")
        print("=" * 108)
        print(f"Mode         : {self.cfg.mode}")
        print(f"Dataset      : {DATASET_OUTPUT_NAMES[self.cfg.dataset]}")
        print(f"Input root   : {self.cfg.input_root}")
        print(f"Images       : {len(files)}")
        print(f"Output root  : {self.cfg.output_root}")
        print("DOD steps    : 1 (official pretrained contract)")
        print(f"DOD timestep : {DOD_NATIVE_TIMESTEP}")
        print(f"Process size : {self.cfg.process_size}")
        print("Preprocess   : official DOD multiple-of-8 for baseline + hazecdg")
        if self.cfg.mode == "hazecdg":
            print("Sensor pad   : LHD branch only, symmetric latent pad -> crop")
        print(f"Precision    : {self.cfg.mixed_precision}")
        print(f"Seed policy  : fixed seed={self.cfg.seed} reset for every image")
        print(f"Stage1       : {self.cfg.dod_stage1_checkpoint}")
        print(f"Stage2       : {self.cfg.dod_stage2_checkpoint}")
        if self.cfg.mode == "hazecdg":
            print(f"G_D weight   : {self.cfg.lhd_stage1_checkpoint}")
            print("HazeCDG calls   : 1 exact counter-degradation correction before DOD sched.step")
            print("Sparse hold  : N/A (DOD has one restoration step)")
            print("Guidance rule: q=2g (parameter-free)")
        print("=" * 108)

        total = len(files)
        for idx, image_path in enumerate(files, 1):
            relative_name = image_path.relative_to(self.cfg.input_root).as_posix()
            out_path = self._output_path(image_path)
            if (
                self.cfg.skip_existing
                and out_path.is_file()
                and out_path.stat().st_size > 0
            ):
                print(f"[{idx:04d}/{total:04d}] [skip] {relative_name} -> {out_path}")
                continue

            print(f"\n[{idx:04d}/{total:04d}] {relative_name} | seed={self.cfg.seed}")
            try:
                result, meta = self._run_one(image_path, self.cfg.seed)
                self._save_tensor_atomic(result, out_path)
                print(
                    f"[save] {out_path} | work={meta['work_w']}x{meta['work_h']} "
                    f"| restored={meta['orig_w']}x{meta['orig_h']}"
                )
                del result
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA OOM for {relative_name}. DOD already uses latent tiling; "
                    "reduce runtime.latent_tiled_size only if needed."
                )

        print("\n[done] DOD baseline/HazeCDG inference completed.")


# =============================================================================
# Config construction
# =============================================================================


def build_runtime_config(
    project_root: Path,
    config_path: Path,
    cfg_dict: Dict[str, Any],
) -> RuntimeConfig:
    del config_path  # retained in signature for parity with the LHD runner
    mode = normalize_mode(nested_get(cfg_dict, ["experiment", "mode"], "hazecdg"))

    dod_repo_root = resolve_path(
        project_root,
        nested_get(cfg_dict, ["paths", "dod_repo_root"], "./third_party/DOD"),
    )
    dod_sd21_root = resolve_path(
        project_root,
        nested_get(cfg_dict, ["paths", "dod_sd21_root"], "./checkpoints/DOD/sd21"),
    )
    dod_stage1_checkpoint = resolve_path(
        project_root,
        nested_get(cfg_dict, ["paths", "dod_stage1_checkpoint"], "./checkpoints/DOD/stage1.pkl"),
    )
    dod_stage2_checkpoint = resolve_path(
        project_root,
        nested_get(cfg_dict, ["paths", "dod_stage2_checkpoint"], "./checkpoints/DOD/stage2.pkl"),
    )
    dod_mfm_checkpoint = resolve_path(
        project_root,
        nested_get(
            cfg_dict,
            ["paths", "dod_mfm_checkpoint"],
            "./checkpoints/DOD/mfm/256x256_diffusion_uncond.pt",
        ),
    )
    dod_prompt_embeds = resolve_path(
        project_root,
        nested_get(
            cfg_dict,
            ["paths", "dod_prompt_embeds"],
            "./third_party/DOD/prompt_embeds.pt",
        ),
    )

    _require_dir(dod_repo_root, "DOD repo")
    _require_dir(dod_sd21_root, "DOD SD2.1 Diffusers directory")
    for sub in ["tokenizer", "text_encoder", "scheduler", "vae", "unet"]:
        _require_dir(dod_sd21_root / sub, f"DOD SD2.1/{sub}")
    _require_file(dod_stage1_checkpoint, "DOD Stage1 checkpoint")
    _require_file(dod_stage2_checkpoint, "DOD Stage2 checkpoint")
    _require_file(dod_mfm_checkpoint, "DOD MFM DDPM checkpoint")
    _require_file(dod_prompt_embeds, "DOD prompt embeddings")

    lhd_repo_root = resolve_path(
        project_root,
        nested_get(
            cfg_dict,
            ["paths", "lhd_repo_root"],
            "./third_party/Learning-Hazing-to-Dehazing",
        ),
    )
    lhd_stage1_config = resolve_path(
        project_root,
        nested_get(
            cfg_dict,
            ["paths", "lhd_stage1_config"],
            "./third_party/Learning-Hazing-to-Dehazing/configs/inference/stage1.yaml",
        ),
    )
    lhd_sd_checkpoint = resolve_path(
        project_root,
        nested_get(
            cfg_dict,
            ["paths", "lhd_sd_checkpoint"],
            "./checkpoints/Learning-Hazing-to-Dehazing/v2-1_512-ema-pruned.ckpt",
        ),
    )
    lhd_stage1_checkpoint = resolve_path(
        project_root,
        nested_get(
            cfg_dict,
            ["paths", "lhd_stage1_checkpoint"],
            "./checkpoints/Learning-Hazing-to-Dehazing/stage1.pt",
        ),
    )
    if mode == "hazecdg":
        _require_dir(lhd_repo_root, "LHD repo")
        _require_file(lhd_stage1_config, "LHD Stage1 config")
        _require_file(lhd_sd_checkpoint, "LHD SD2.1 checkpoint")
        _require_file(lhd_stage1_checkpoint, "LHD Stage1 HazeGen checkpoint")

    dataset = normalize_dataset_name(nested_get(cfg_dict, ["data", "dataset"], "RTTS"))
    dataset_root = resolve_path(
        project_root,
        nested_get(cfg_dict, ["data", "dataset_root"], "./datasets/dehaze"),
    )
    _require_dir(dataset_root, "data.dataset_root")
    input_root = resolve_dataset_input_root(
        dataset_root=dataset_root,
        dataset=dataset,
        explicit_input_root=nested_get(cfg_dict, ["data", "input_root"], ""),
        project_root=project_root,
    )

    configured_output = str(nested_get(cfg_dict, ["data", "output_root"], "") or "").strip()
    output_root = (
        resolve_path(project_root, configured_output)
        if configured_output
        else default_output_root(project_root, mode, dataset).resolve()
    )

    max_images = int(nested_get(cfg_dict, ["data", "max_images"], 0))
    if max_images < 0:
        raise ValueError("data.max_images must be >= 0 (0 means full dataset).")

    steps = validate_inference_steps(nested_get(cfg_dict, ["inference", "steps"], 1))
    process_size = int(nested_get(cfg_dict, ["inference", "process_size"], 256))
    if process_size <= 0:
        raise ValueError("inference.process_size must be positive.")
    upscale = int(nested_get(cfg_dict, ["inference", "upscale"], 1))
    if upscale <= 0:
        raise ValueError("inference.upscale must be a positive integer.")

    mixed_precision = str(
        nested_get(cfg_dict, ["inference", "mixed_precision"], "fp16")
    ).strip().lower()
    if mixed_precision not in {"fp16", "fp32"}:
        raise ValueError("inference.mixed_precision must be fp16 or fp32.")

    latent_tiled_size = int(
        nested_get(cfg_dict, ["inference", "latent_tiled_size"], 96)
    )
    latent_tiled_overlap = int(
        nested_get(cfg_dict, ["inference", "latent_tiled_overlap"], 32)
    )
    if latent_tiled_size <= 0:
        raise ValueError("inference.latent_tiled_size must be positive.")
    if not (0 <= latent_tiled_overlap < latent_tiled_size):
        raise ValueError(
            "inference.latent_tiled_overlap must satisfy 0 <= overlap < tile_size."
        )

    return RuntimeConfig(
        project_root=project_root,
        dod_repo_root=dod_repo_root,
        dod_sd21_root=dod_sd21_root,
        dod_stage1_checkpoint=dod_stage1_checkpoint,
        dod_stage2_checkpoint=dod_stage2_checkpoint,
        dod_mfm_checkpoint=dod_mfm_checkpoint,
        dod_prompt_embeds=dod_prompt_embeds,
        lhd_repo_root=lhd_repo_root,
        lhd_stage1_config=lhd_stage1_config,
        lhd_sd_checkpoint=lhd_sd_checkpoint,
        lhd_stage1_checkpoint=lhd_stage1_checkpoint,
        dataset=dataset,
        dataset_root=dataset_root,
        input_root=input_root,
        output_root=output_root,
        mode=mode,
        device=str(nested_get(cfg_dict, ["runtime", "device"], "cuda:0")),
        expected_conda_env=str(
            nested_get(
                cfg_dict,
                ["runtime", "expected_conda_env"],
                "LHD_xformers_blackwell",
            )
        ),
        lhd_attention_mode=str(
            nested_get(cfg_dict, ["runtime", "lhd_attention_mode"], "xformers")
        ).strip().lower(),
        cuda_visible_devices=str(
            nested_get(cfg_dict, ["runtime", "cuda_visible_devices"], "") or ""
        ),
        seed=int(nested_get(cfg_dict, ["runtime", "seed"], 231)),
        max_images=max_images,
        skip_existing=bool(nested_get(cfg_dict, ["data", "skip_existing"], True)),
        steps=steps,
        process_size=process_size,
        upscale=upscale,
        mixed_precision=mixed_precision,
        lora_rank=int(nested_get(cfg_dict, ["inference", "lora_rank"], 8)),
        latent_tiled_size=latent_tiled_size,
        latent_tiled_overlap=latent_tiled_overlap,
        vae_decoder_tiled_size=int(
            nested_get(cfg_dict, ["inference", "vae_decoder_tiled_size"], 224)
        ),
        vae_encoder_tiled_size=int(
            nested_get(cfg_dict, ["inference", "vae_encoder_tiled_size"], 1024)
        ),
        degradation_prompt=str(
            nested_get(
                cfg_dict,
                ["hazecdg", "degradation_prompt"],
                "hazy, foggy, misty, obscure, smoggy.",
            )
        ),
        save_original_size=bool(
            nested_get(cfg_dict, ["runtime", "save_original_size"], True)
        ),
        log_diagnostics=bool(
            nested_get(cfg_dict, ["hazecdg", "log_diagnostics"], True)
        ),
        raw_cfg=cfg_dict,
    )


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DOD official one-step baseline / HazeCDG exact counter-degradation guidance."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/eval_DOD_HazeCDG.yaml",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate paths/environment and load required model(s), but do not infer.",
    )
    args = parser.parse_args()

    root = project_root_from_script()
    config_path = resolve_path(root, args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg_dict = load_yaml(config_path)
    runtime_cfg = build_runtime_config(root, config_path, cfg_dict)
    runner = DODRunner(runtime_cfg)

    if args.check_only:
        files = list_images(runtime_cfg.input_root)
        if runtime_cfg.max_images > 0:
            files = files[: runtime_cfg.max_images]
        print("=" * 108)
        print("[check] DOD/HazeCDG configuration loaded.")
        print(f"[check] mode            : {runtime_cfg.mode}")
        print(f"[check] dataset         : {DATASET_OUTPUT_NAMES[runtime_cfg.dataset]}")
        print(f"[check] input images    : {len(files)}")
        print("[check] DOD steps       : 1")
        print(f"[check] native timestep : {DOD_NATIVE_TIMESTEP}")
        print(f"[check] Stage1          : {runtime_cfg.dod_stage1_checkpoint}")
        print(f"[check] Stage2          : {runtime_cfg.dod_stage2_checkpoint}")
        print(f"[check] MFM             : {runtime_cfg.dod_mfm_checkpoint}")
        print(f"[check] SD2.1           : {runtime_cfg.dod_sd21_root}")
        if runtime_cfg.mode == "hazecdg":
            print("[check] HazeCDG rule    : eps_R - 2g")
            print("[check] guidance calls  : exactly 1")
            print("[check] insertion       : after DOD model_pred, before sched.step")
            print("[check] sparse hold     : N/A for one-step DOD")
            print("[check] LHD compatibility: verified during model loading")
        print("[check] Inference was not started.")
        print("=" * 108)
        return

    runner.run()


if __name__ == "__main__":
    main()
