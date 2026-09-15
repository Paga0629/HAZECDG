#!/usr/bin/env python3
"""
Learning-Hazing-to-Dehazing runner with Haze Counter-Degradation Guidance (HazeCDG).

Intended project location
-------------------------
    scripts/eval_LHD_HazeCDG.py

Method summary
--------------
HazeCDG reuses the frozen haze-generation model G_D from LHD during inference
of the frozen restoration model G_R. At a sensing anchor a, both degradation
predictions are evaluated at the same restoration state x_a and timestep:

    g_a = eps_D,a - eps_base,a

where eps_D,a is the prediction with the learned IRControlNet branch enabled
and eps_base,a is the prediction of the same frozen latent-diffusion backbone
with that branch disabled. The resulting haze differential g_a is interpreted
as the local change in diffusion prediction induced by the learned degradation
branch.

Reversing this local change defines the counter-degradation reference

    eps_CD,a = eps_base,a - g_a.

HazeCDG preserves the relative prediction offset between the restoration and
degradation models while replacing eps_D,a by eps_CD,a. This yields the final
parameter-free guidance rule used in the paper:

    eps_R^HazeCDG = eps_R - 2 g_a.

The factor 2 is therefore not a tunable guidance strength; it follows from the
reference change from eps_base + g_a to eps_base - g_a.

Sparse degradation sensing
--------------------------
For the 50-step DiffDehaze experiments, HazeCDG uses fixed, deterministic,
nested sensing schedules satisfying

    A5 subset A10 subset A25 subset A50.

At an anchor, the current haze differential is recomputed. Between anchors,
no additional G_D / eps_base forward is performed; the most recently sensed
haze differential is causally reused. The implementation caches its equivalent
2*g_a correction for efficiency, which is algebraically identical to the paper's
Algorithm 1 update eps_R - 2*g_hat.

When K=50, every reverse step is a sensing anchor and the implementation reduces
to dense HazeCDG. K=5 is the default sparse setting reported in the paper.

Supported evaluation datasets
-----------------------------
Only the three real-world haze benchmarks used in the paper are supported:
RTTS, URHI, and Fattal. Expected public data layout:

    dataset/RTTS/
    dataset/URHI/
    dataset/Fattal/

The code evaluates every image present in the selected folder. To reproduce the
paper protocol, place the 4,322 RTTS images, the fixed 150-image URHI subset,
and the 31 Fattal images in the corresponding folders.

Controlled modes
----------------
    baseline
        Official LHD DiffDehaze (Stage2) with the standard SpacedSampler.
        Stage1 HazeGen is not loaded.

    hazecdg
        The same DiffDehaze trajectory with HazeCDG inserted between the
        restoration noise prediction and the unchanged sampler update.

Implementation scope
--------------------
- G_R and G_D remain frozen; no model parameter is updated.
- No test-time parameter adaptation is performed.
- HazeCDG introduces no manually tuned guidance-strength hyperparameter.
- Standard LHD SpacedSampler is used; AccSamp / RGB fidelity guidance are off.
- Stage1 uses the zero image hint for degradation sensing, consistent with the
  real-image HazeGen setting described in the paper.

Preprocessing
-------------
The controlled DiffDehaze preprocessing used for evaluation is preserved:
1) if either input dimension is below 512, torchvision Resize(512) is applied,
2) H and W are resized independently to the nearest multiple of 64,
3) no zero padding is used,
4) the saved output is restored to the original image resolution.

Output folders
--------------
    baseline : outputs/LHD/baseline/<DATASET>/
    hazecdg  : outputs/LHD/hazecdg/<DATASET>/K005, K010, K025, K050, ...
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_OUTPUT_NAME = "LHD"
NUMERICAL_EPS = 1e-12
SUPPORTED_MODES = ("baseline", "hazecdg")

DATASET_ALIASES = {
    "rtts": "rtts",
    "urhi": "urhi",
    "fattal": "fattal",
}

DATASET_OUTPUT_NAMES = {
    "rtts": "RTTS",
    "urhi": "URHI",
    "fattal": "Fattal",
}

DATASET_RELATIVE_DIRS = {
    "rtts": Path("RTTS"),
    "urhi": Path("URHI"),
    "fattal": Path("Fattal"),
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
    # scripts/eval_LHD_HazeCDG.py -> project root
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


def torch_load_compat(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap_state_dict(obj: Any) -> Any:
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def normalize_oom_retry_scales(values: Sequence[float]) -> List[float]:
    out = [float(v) for v in values]
    if not out:
        return []
    if any((not np.isfinite(v)) or v <= 0.0 or v >= 1.0 for v in out):
        raise ValueError("runtime.oom_retry_scales must contain values in (0,1).")
    if any(out[i] <= out[i + 1] for i in range(len(out) - 1)):
        raise ValueError("runtime.oom_retry_scales must be strictly descending.")
    return out


def nearest_multiple_work_size(
    orig_h: int,
    orig_w: int,
    multiple: int = 64,
) -> Tuple[int, int]:
    h, w = int(orig_h), int(orig_w)
    multiple = int(multiple)
    if h <= 0 or w <= 0 or multiple <= 0:
        raise ValueError("Image sizes and multiple must be positive.")

    def _nearest(x: int) -> int:
        return max(multiple, int(round(float(x) / float(multiple))) * multiple)

    return _nearest(h), _nearest(w)


# =============================================================================
# Parameter-free guidance cores
# =============================================================================


def apply_hazecdg_guidance(
    eps_r: torch.Tensor,
    haze_differential: torch.Tensor,
    *,
    eps: float = NUMERICAL_EPS,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Apply HazeCDG in noise-prediction space.

    The paper defines the haze differential and counter-degradation reference as

        g_t      = eps_D - eps_base,
        eps_CD   = eps_base - g_t.

    Preserving the restoration-degradation prediction offset while replacing
    eps_D = eps_base + g_t by eps_CD gives

        eps_R^HazeCDG = eps_R - 2 g_t.

    The factor 2 follows from the counter-degradation construction and is not a
    tuned guidance scale.
    """
    if eps_r.shape != haze_differential.shape:
        raise ValueError(
            "HazeCDG guidance shape mismatch: "
            f"eps_r={tuple(eps_r.shape)}, "
            f"g={tuple(haze_differential.shape)}"
        )

    g = haze_differential.detach().float()
    g_norm = float(torch.linalg.vector_norm(g.reshape(-1)).item())

    if (not np.isfinite(g_norm)) or g_norm <= eps:
        correction = torch.zeros_like(g)
        guided = eps_r
    else:
        correction = 2.0 * g
        guided = (eps_r.float() - correction).to(dtype=eps_r.dtype)

    diag = {
        "g_norm": g_norm,
        "correction_norm": float(
            torch.linalg.vector_norm(correction.reshape(-1)).item()
        ),
        "guidance_factor": 2.0,
    }
    return guided, diag


def choose_anchor_eval_indices(
    *,
    steps: int,
    anchor_count: int,
) -> List[int]:
    """Choose deterministic causal sensing anchors over the reverse calls.

    The paper experiments use fixed nested schedules for the standard 50-step
    DiffDehaze trajectory:

        A5 subset A10 subset A25 subset A50.

    Increasing K therefore adds sensing anchors while preserving every anchor
    from the smaller-K schedule. These exact indices are kept here to reproduce
    the reported experiments. For non-paper settings, a deterministic evenly
    spaced fallback is used.
    """

    steps = int(steps)
    anchor_count = int(anchor_count)

    if steps <= 0:
        raise ValueError("steps must be positive.")
    if anchor_count <= 0:
        raise ValueError("hazecdg.anchor_count must be positive.")
    if anchor_count > steps:
        raise ValueError(
            f"hazecdg.anchor_count={anchor_count} exceeds sampler steps={steps}."
        )

    # ------------------------------------------------------------------
    # Exact fixed nested schedules used for the 50-step HazeCDG experiments.
    #
    # A5 ⊂ A10 ⊂ A25 ⊂ A50
    #
    # Each larger K preserves every anchor used by the smaller K setting.
    # ------------------------------------------------------------------
    if steps == 50:
        nested_schedules = {
            5: [
                0, 12, 24, 37, 49,
            ],
            10: [
                0, 6, 12, 18, 24,
                28, 32, 37, 43, 49,
            ],
            25: [
                0, 2, 4, 6, 8,
                10, 12, 14, 16, 18,
                20, 22, 24, 26, 28,
                30, 32, 34, 37, 39,
                41, 43, 45, 47, 49,
            ],
            50: list(range(50)),
        }

        if anchor_count in nested_schedules:
            anchors = list(nested_schedules[anchor_count])

            if len(anchors) != anchor_count:
                raise RuntimeError(
                    f"Nested schedule length mismatch for K={anchor_count}."
                )
            if anchors != sorted(set(anchors)):
                raise RuntimeError(
                    f"Nested schedule contains duplicate/non-monotonic anchors: {anchors}"
                )
            if anchors[0] != 0 or anchors[-1] != steps - 1:
                raise RuntimeError(
                    f"Nested schedule must include first/final calls: {anchors}"
                )

            return anchors

    # ------------------------------------------------------------------
    # Generic fallback for non-paper settings / arbitrary K.
    # ------------------------------------------------------------------
    if anchor_count == steps:
        return list(range(steps))
    if anchor_count == 1:
        return [0]

    raw = np.rint(
        np.linspace(0, steps - 1, num=anchor_count, dtype=np.float64)
    ).astype(np.int64)

    out: List[int] = []
    for value in raw.tolist():
        idx = int(value)
        if out and idx <= out[-1]:
            idx = out[-1] + 1
        if idx >= steps:
            break
        out.append(idx)

    if len(out) != anchor_count:
        raise RuntimeError(
            f"Failed to construct {anchor_count} unique anchors over {steps} steps."
        )

    return out




# =============================================================================
# Environment and official LHD imports
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
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"runtime.device={device} but CUDA is not available.")
        index = torch.device(device).index or 0
        print(f"[env] GPU               : {torch.cuda.get_device_name(index)}")


def configure_attention_backend(mode: str) -> None:
    key = str(mode or "sdp").strip().lower()
    if key not in {"sdp", "xformers", "vanilla"}:
        raise ValueError("runtime.attention_mode must be one of: sdp | xformers | vanilla")
    os.environ["ATTN_MODE"] = key
    print(f"[env] Attention backend : {key}")


def verify_xformers_runtime(device):
    import torch

    try:
        import xformers
        from xformers.ops import memory_efficient_attention
    except Exception as exc:
        raise RuntimeError(
            f"xFormers could not be imported: {exc}"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device(device)

    if device.type != "cuda":
        return

    try:
        q = torch.randn(
            1,
            64,
            8,
            64,
            device=device,
            dtype=torch.float16,
        )

        with torch.inference_mode():
            _ = memory_efficient_attention(q, q, q)

        torch.cuda.synchronize(device)

    except Exception as exc:
        raise RuntimeError(
            "xFormers memory-efficient attention failed on the active GPU.\n"
            f"GPU: {torch.cuda.get_device_name(device)}\n"
            f"Capability: {torch.cuda.get_device_capability(device)}\n"
            f"xFormers: {xformers.__version__}\n"
            f"Original error: {exc}"
        ) from exc


def import_lhd(repo_root: Path):
    repo_root = repo_root.resolve()
    required = [
        repo_root / "inference_stage2.py",
        repo_root / "diffbir" / "model" / "cldm.py",
        repo_root / "diffbir" / "model" / "gaussian_diffusion.py",
        repo_root / "diffbir" / "sampler" / "spaced_sampler.py",
        repo_root / "configs" / "inference" / "stage1.yaml",
        repo_root / "configs" / "inference" / "stage2.yaml",
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
        from diffbir.model import ControlLDM, Diffusion
        from diffbir.sampler import SpacedSampler
        from diffbir.utils.common import instantiate_from_config
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import official Learning-Hazing-to-Dehazing runtime from {repo_root}."
        ) from exc

    print(f"[env] LHD repo           : {repo_root}")
    print("[env] LHD imports        : OK")
    return OmegaConf, ControlLDM, Diffusion, SpacedSampler, instantiate_from_config

def patch_lhd_vae_attention() -> None:
    """
    Keep the official LHD repository untouched.

    When the global LHD attention backend is xFormers, use the official
    PyTorch SDPA implementation only for VAE attention. Diffusion UNet /
    cross-attention remains on xFormers.
    """
    import diffbir.model.vae as lhd_vae

    # Avoid patching more than once.
    if getattr(lhd_vae, "_hazecdg_vae_attention_patched", False):
        return

    original_make_attn = lhd_vae.make_attn

    def make_attn_hazecdg(
        in_channels,
        attn_type="vanilla",
        attn_kwargs=None,
    ):
        if attn_type == "xformers":
            attn_type = "sdp"

        return original_make_attn(
            in_channels,
            attn_type=attn_type,
            attn_kwargs=attn_kwargs,
        )

    lhd_vae.make_attn = make_attn_hazecdg
    lhd_vae._hazecdg_vae_attention_patched = True

    print("[env] LHD VAE attention  : sdp (xFormers kept for UNet)")

def patch_lhd_xformers_attention_dtype() -> None:
    """
    Keep the official LHD repository unchanged.

    Cast only xFormers Q/K/V tensors to FP16 on CUDA, then cast the
    attention output back to the original dtype before the output projection.
    """
    import torch
    from einops import rearrange
    import diffbir.model.attention as lhd_attention
    from diffbir.model.config import Config

    cls = lhd_attention.MemoryEfficientCrossAttention

    if getattr(cls, "_hazecdg_dtype_patched", False):
        return

    def forward_hazecdg(self, x, context=None, mask=None):
        q = self.to_q(x)

        if context is None:
            context = x

        k = self.to_k(context)
        v = self.to_v(context)

        original_dtype = q.dtype
        b, _, _ = q.shape

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        if q.is_cuda and q.dtype == torch.float32:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        out = Config.xformers.ops.memory_efficient_attention(
            q,
            k,
            v,
            attn_bias=None,
            op=self.attention_op,
        )

        out = out.to(original_dtype)

        if mask is not None:
            raise NotImplementedError

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )

        return self.to_out(out)

    cls.forward = forward_hazecdg
    cls._hazecdg_dtype_patched = True

    print("[env] LHD xFormers QKV    : fp16")

# =============================================================================
# Model loading
# =============================================================================


def load_official_lhd_model(
    *,
    config_path: Path,
    sd_checkpoint: Path,
    controlnet_checkpoint: Path,
    device: torch.device,
    OmegaConf,
    instantiate_from_config,
    label: str,
):
    cfg = OmegaConf.load(str(config_path))
    model = instantiate_from_config(cfg.model.cldm)

    sd_blob = torch_load_compat(sd_checkpoint)
    sd_state = unwrap_state_dict(sd_blob)
    if not isinstance(sd_state, dict):
        raise RuntimeError(f"Unexpected SD checkpoint format: {sd_checkpoint}")
    unused, missing = model.load_pretrained_sd(sd_state)
    print(f"[env] {label} SD          : {sd_checkpoint}")
    print(f"[env] {label} SD missing  : {len(missing)}")
    print(f"[env] {label} SD unused   : {len(unused)}")

    control_blob = torch_load_compat(controlnet_checkpoint)
    control_state = unwrap_state_dict(control_blob)
    if not isinstance(control_state, dict):
        raise RuntimeError(f"Unexpected ControlNet checkpoint format: {controlnet_checkpoint}")
    model.load_controlnet_from_ckpt(control_state)
    print(f"[env] {label} ControlNet  : {controlnet_checkpoint}")

    model.eval().to(device)
    model.requires_grad_(False)

    diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.eval().to(device)
    diffusion.requires_grad_(False)
    return model, diffusion


def load_official_stage1_controlnet_only(
    *,
    config_path: Path,
    sd_checkpoint: Path,
    controlnet_checkpoint: Path,
    device: torch.device,
    OmegaConf,
    instantiate_from_config,
):
    """Load exact Stage1 weights but keep only Stage1 ControlNet on GPU."""
    cfg = OmegaConf.load(str(config_path))
    model = instantiate_from_config(cfg.model.cldm)

    sd_blob = torch_load_compat(sd_checkpoint)
    sd_state = unwrap_state_dict(sd_blob)
    if not isinstance(sd_state, dict):
        raise RuntimeError(f"Unexpected SD checkpoint format: {sd_checkpoint}")
    unused, missing = model.load_pretrained_sd(sd_state)
    print(f"[env] G_D/stage1 SD          : {sd_checkpoint}")
    print(f"[env] G_D/stage1 SD missing  : {len(missing)}")
    print(f"[env] G_D/stage1 SD unused   : {len(unused)}")

    control_blob = torch_load_compat(controlnet_checkpoint)
    control_state = unwrap_state_dict(control_blob)
    if not isinstance(control_state, dict):
        raise RuntimeError(f"Unexpected ControlNet checkpoint format: {controlnet_checkpoint}")
    model.load_controlnet_from_ckpt(control_state)
    print(f"[env] G_D/stage1 ControlNet  : {controlnet_checkpoint}")

    if not hasattr(model, "controlnet") or model.controlnet is None:
        raise RuntimeError("Official Stage1 ControlLDM has no `controlnet` module.")

    scale_factor = float(model.scale_factor)
    diffusion = instantiate_from_config(cfg.model.diffusion)
    stage1_betas = np.asarray(diffusion.betas, dtype=np.float64).copy()

    controlnet = model.controlnet
    model.controlnet = None
    controlnet.eval().requires_grad_(False)
    controlnet.to(device)

    del diffusion, model, control_blob, control_state, sd_blob, sd_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[HazeCDG] Stage1 VRAM policy : ControlNet only")
    print("[HazeCDG] Stage1 SD/VAE/CLIP : not resident on GPU")
    return controlnet, scale_factor, stage1_betas


# =============================================================================
# Sparse HazeCDG sampler
# =============================================================================


def make_sparse_guidance_sampler_class(OfficialSpacedSampler):
    from diffbir.model.util import timestep_embedding

    class SparseHazeCDGSampler(OfficialSpacedSampler):
        def __init__(
            self,
            *args,
            stage1_controlnet,
            degradation_text_condition: torch.Tensor,
            anchor_count: int,
            log_every: int = 10,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)

            if self.parameterization != "eps":
                raise ValueError(
                    "Sparse HazeCDG is defined in epsilon space, "
                    f"but parameterization={self.parameterization!r}."
                )
            if stage1_controlnet is None:
                raise ValueError("Sparse HazeCDG requires official Stage1 HazeGen ControlNet.")
            if degradation_text_condition is None:
                raise ValueError("Sparse HazeCDG requires the degradation text condition.")
            if int(anchor_count) <= 0:
                raise ValueError("HazeCDG anchor_count must be positive.")

            self.stage1_controlnet = stage1_controlnet
            self.degradation_text_condition = degradation_text_condition.detach()
            self.anchor_count = int(anchor_count)
            self.log_every = max(0, int(log_every))
            self.degradation_condition_hint: Optional[torch.Tensor] = None
            self.reset_image_state()

        def reset_image_state(self) -> None:
            self.anchor_eval_indices: List[int] = []
            self.anchor_eval_set = set()
            self.call_index = 0
            self.query_count = 0
            self.guided_count = 0
            self.reused_step_count = 0
            self.cached_counter_degradation_correction: Optional[torch.Tensor] = None
            self.cached_anchor_call_index: Optional[int] = None

            self.sum_anchor_g_norm = 0.0
            self.sum_anchor_correction_norm = 0.0
            self.sum_applied_correction_norm = 0.0

        def set_anchor_count(self, anchor_count: int) -> None:
            anchor_count = int(anchor_count)
            if anchor_count <= 0:
                raise ValueError("HazeCDG anchor_count must be >= 1.")
            self.anchor_count = anchor_count

        def set_degradation_condition_hint(self, reference_c_img: torch.Tensor) -> None:
            self.degradation_condition_hint = torch.zeros_like(reference_c_img)

        @staticmethod
        def _decode_from_shared_unet_trunk(
            unet,
            *,
            x_dtype: torch.dtype,
            h_middle: torch.Tensor,
            encoder_skips: Sequence[torch.Tensor],
            emb: torch.Tensor,
            context: torch.Tensor,
            control: Optional[Sequence[torch.Tensor]],
        ) -> torch.Tensor:
            hs = list(encoder_skips)
            if control is None:
                control_work = None
                h = h_middle
            else:
                control_work = list(control)
                expected = len(encoder_skips) + 1
                if len(control_work) != expected:
                    raise RuntimeError(
                        "Unexpected ControlNet feature count: "
                        f"got {len(control_work)}, expected {expected}."
                    )
                h = h_middle + control_work.pop()

            for module in unet.output_blocks:
                if not hs:
                    raise RuntimeError("UNet decoder requested too many skip tensors.")
                skip = hs.pop()
                if control_work is not None:
                    if not control_work:
                        raise RuntimeError(
                            "Stage1 ControlNet residuals exhausted before decoder finished."
                        )
                    skip = skip + control_work.pop()
                h = torch.cat([h, skip], dim=1)
                h = module(h, emb, context)

            if hs:
                raise RuntimeError(f"Unused shared UNet skip tensors remain: {len(hs)}")
            if control_work is not None and control_work:
                raise RuntimeError(
                    f"Unused Stage1 ControlNet residuals remain: {len(control_work)}"
                )
            h = h.type(x_dtype)
            return unet.out(h)

        @torch.no_grad()
        def _shared_degradation_base_eps_exact(
            self,
            shared_model,
            x: torch.Tensor,
            model_t: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if self.degradation_condition_hint is None:
                raise RuntimeError("Stage1 zero hint has not been initialized for this image.")

            control_deg = self.stage1_controlnet(
                x=x,
                hint=self.degradation_condition_hint,
                timesteps=model_t,
                context=self.degradation_text_condition,
            )
            control_deg = [c * 1.0 for c in control_deg]

            unet = shared_model.unet
            t_emb = timestep_embedding(model_t, unet.model_channels, repeat_only=False)
            emb = unet.time_embed(t_emb)
            h = x.type(unet.dtype)
            emb = emb.type(unet.dtype)
            context = self.degradation_text_condition.type(unet.dtype)
            encoder_skips = []
            for module in unet.input_blocks:
                h = module(h, emb, context)
                encoder_skips.append(h)
            h_middle = unet.middle_block(h, emb, context)

            eps_deg = self._decode_from_shared_unet_trunk(
                unet,
                x_dtype=x.dtype,
                h_middle=h_middle,
                encoder_skips=encoder_skips,
                emb=emb,
                context=context,
                control=control_deg,
            )
            del control_deg

            eps_base = self._decode_from_shared_unet_trunk(
                unet,
                x_dtype=x.dtype,
                h_middle=h_middle,
                encoder_skips=encoder_skips,
                emb=emb,
                context=context,
                control=None,
            )
            return eps_deg, eps_base

        @staticmethod
        def _restoration_eps(
            model,
            x: torch.Tensor,
            model_t: torch.Tensor,
            cond: Dict[str, torch.Tensor],
            uncond: Optional[Dict[str, torch.Tensor]],
            cfg_scale: float,
        ) -> torch.Tensor:
            if uncond is None or cfg_scale == 1.0:
                return model(x, model_t, cond)
            eps_cond = model(x, model_t, cond)
            eps_uncond = model(x, model_t, uncond)
            return eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        def _accumulate_anchor_diag(self, diag: Dict[str, float]) -> None:
            self.query_count += 1
            self.sum_anchor_g_norm += diag["g_norm"]
            self.sum_anchor_correction_norm += diag["correction_norm"]

        def _maybe_log_application(
            self,
            *,
            model_t: torch.Tensor,
            is_anchor: bool,
            correction: torch.Tensor,
            diag: Optional[Dict[str, float]],
        ) -> None:
            if self.log_every <= 0:
                return
            step_num = self.call_index + 1
            if not (step_num == 1 or step_num % self.log_every == 0):
                return
            corr_norm = float(torch.linalg.vector_norm(correction.float()).item())
            if is_anchor and diag is not None:
                extra = (
                    f" | anchor=1"
                    f" | guidance=-2g"
                    f" | ||g||={diag['g_norm']:.4e}"
                )
            else:
                extra = (
                    f" | anchor=0"
                    f" | reuse_from={self.cached_anchor_call_index + 1 if self.cached_anchor_call_index is not None else -1}"
                    f" | reuse=2*g_from_anchor"
                )
            print(
                f"[HazeCDG] step={step_num:02d}"
                f" | native_t={int(model_t[0].item())}"
                f" | ||q||={corr_norm:.4e}"
                f"{extra}"
            )

        @torch.no_grad()
        def p_sample(
            self,
            model,
            x: torch.Tensor,
            model_t: torch.Tensor,
            t: torch.Tensor,
            cond: Dict[str, torch.Tensor],
            uncond: Optional[Dict[str, torch.Tensor]],
            cfg_scale: float,
        ) -> torch.Tensor:
            if x.shape[0] != 1:
                raise RuntimeError(
                    "Sparse HazeCDG currently supports batch-size-1 only; "
                    f"got batch={x.shape[0]}."
                )

            # 1) The ordinary Stage2 restoration prediction is required by the
            #    baseline sampler at every reverse step.
            eps_r = self._restoration_eps(model, x, model_t, cond, uncond, cfg_scale)
            is_anchor = self.call_index in self.anchor_eval_set
            diag: Optional[Dict[str, float]] = None

            if is_anchor:
                # 2a) Sense the local haze differential and apply Eq. (4):
                #
                #         g_a              = eps_D,a - eps_base,a
                #         eps_CD,a         = eps_base,a - g_a
                #         eps_R^HazeCDG,a  = eps_R,a - 2 g_a
                #
                #     The -2g_a update follows from replacing the degradation
                #     reference eps_base+g_a by eps_base-g_a while preserving
                #     the restoration-degradation prediction offset.
                eps_deg, eps_base = self._shared_degradation_base_eps_exact(
                    model, x, model_t
                )
                g_t = eps_deg - eps_base
                eps_guided, diag = apply_hazecdg_guidance(eps_r, g_t)

                anchor_correction = (
                    eps_r.detach().float() - eps_guided.detach().float()
                ).contiguous()
                self.cached_counter_degradation_correction = anchor_correction
                self.cached_anchor_call_index = self.call_index
                correction = anchor_correction
                self._accumulate_anchor_diag(diag)
                del eps_deg, eps_base, g_t
            else:
                # 2b) Sparse causal reuse. Algorithm 1 reuses the most recent
                #     haze differential g_hat until the next sensing anchor.
                #     This implementation caches the equivalent 2*g_hat term,
                #     so no G_D or eps_base forward is required here.
                if (
                    self.cached_counter_degradation_correction is None
                    or self.cached_anchor_call_index is None
                ):
                    raise RuntimeError(
                        "Sparse HazeCDG reached a non-anchor before the first sensing anchor."
                    )

                correction = self.cached_counter_degradation_correction
                eps_guided = (
                    eps_r.float() - correction
                ).reshape_as(eps_r.float()).to(dtype=eps_r.dtype)
                self.reused_step_count += 1

            corr_norm = float(torch.linalg.vector_norm(correction.float()).item())
            self.guided_count += 1
            self.sum_applied_correction_norm += corr_norm
            self._maybe_log_application(
                model_t=model_t,
                is_anchor=is_anchor,
                correction=correction,
                diag=diag,
            )

            # 3) Official SpacedSampler DDPM posterior update, changing only eps.
            pred_x0 = self._predict_xstart_from_eps(x, t, eps_guided)
            mean, variance = self.q_posterior_mean_variance(pred_x0, x, t)
            noise = torch.randn_like(x)
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            x_prev = mean + nonzero_mask * torch.sqrt(variance) * noise

            self.call_index += 1
            del eps_r, eps_guided, correction
            return x_prev

        @torch.no_grad()
        def sample(self, *args, **kwargs):
            self.reset_image_state()
            steps = int(kwargs.get("steps", 50))
            self.anchor_eval_indices = choose_anchor_eval_indices(
                steps=steps,
                anchor_count=self.anchor_count,
            )
            self.anchor_eval_set = set(self.anchor_eval_indices)
            print(
                "[HazeCDG] sensing schedule "
                f"| K={self.anchor_count}/{steps} "
                f"| calls={self.anchor_eval_indices}"
            )

            result = super().sample(*args, **kwargs)
            if self.call_index != steps:
                raise RuntimeError(
                    f"Unexpected sampler call count: expected={steps}, got={self.call_index}."
                )
            if self.query_count != self.anchor_count:
                raise RuntimeError(
                    "Sparse degradation query count mismatch: "
                    f"expected={self.anchor_count}, got={self.query_count}."
                )
            return result

    return SparseHazeCDGSampler


# =============================================================================
# Runtime
# =============================================================================


@dataclass
class RuntimeConfig:
    project_root: Path
    repo_root: Path
    stage1_config: Path
    stage2_config: Path
    sd_checkpoint: Path
    stage1_checkpoint: Path
    stage2_checkpoint: Path
    dataset: str
    dataset_root: Path
    input_root: Path
    output_root: Path
    mode: str
    device: str
    expected_conda_env: str
    attention_mode: str
    cuda_visible_devices: str
    seed: int
    max_images: int
    skip_existing: bool
    steps: int
    dehaze_prompt: str
    degradation_prompt: str
    anchor_counts: List[int]
    progress: bool
    log_every: int
    save_original_size: bool
    oom_retry_enabled: bool
    oom_retry_scales: List[float]
    raw_cfg: Dict[str, Any]


class LHDRunner:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        if cfg.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices

        check_conda_environment(cfg.expected_conda_env)
        check_torch_cuda(cfg.device)
        configure_attention_backend(cfg.attention_mode)
        if cfg.attention_mode == "xformers":
            verify_xformers_runtime(cfg.device)

        (
            self.OmegaConf,
            self.ControlLDM,
            self.Diffusion,
            self.OfficialSpacedSampler,
            self.instantiate_from_config,
        ) = import_lhd(cfg.repo_root)

        if cfg.attention_mode == "xformers":
            patch_lhd_vae_attention()
            patch_lhd_xformers_attention_dtype()

        self.device = torch.device(cfg.device)
        self._import_image_runtime()
        self._load_models()
        self._build_sampler()

    def _import_image_runtime(self) -> None:
        try:
            import cv2
            import torchvision
            from torchvision.transforms import InterpolationMode, Resize, ToTensor
        except Exception as exc:
            raise RuntimeError(
                "LHD inference requires cv2 and torchvision in the active environment."
            ) from exc
        self.cv2 = cv2
        self.torchvision = torchvision
        self.ToTensor = ToTensor
        self.Resize = Resize
        self.InterpolationMode = InterpolationMode
        self.rescaler = Resize(
            512,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def _load_models(self) -> None:
        self.gr_model, self.gr_diffusion = load_official_lhd_model(
            config_path=self.cfg.stage2_config,
            sd_checkpoint=self.cfg.sd_checkpoint,
            controlnet_checkpoint=self.cfg.stage2_checkpoint,
            device=self.device,
            OmegaConf=self.OmegaConf,
            instantiate_from_config=self.instantiate_from_config,
            label="G_R/stage2",
        )

        self.stage1_controlnet = None
        self.degradation_text_condition = None
        if self.cfg.mode == "baseline":
            print("[HazeCDG] baseline mode: Stage1 HazeGen is NOT loaded.")
            return

        (
            self.stage1_controlnet,
            stage1_scale_factor,
            stage1_betas,
        ) = load_official_stage1_controlnet_only(
            config_path=self.cfg.stage1_config,
            sd_checkpoint=self.cfg.sd_checkpoint,
            controlnet_checkpoint=self.cfg.stage1_checkpoint,
            device=self.device,
            OmegaConf=self.OmegaConf,
            instantiate_from_config=self.instantiate_from_config,
        )

        if not np.isclose(
            float(self.gr_model.scale_factor),
            float(stage1_scale_factor),
            rtol=1e-8,
            atol=1e-10,
        ):
            raise RuntimeError("Stage1 HazeGen and Stage2 DiffDehaze latent scales differ.")

        gr_betas = np.asarray(self.gr_diffusion.betas, dtype=np.float64)
        if gr_betas.shape != stage1_betas.shape or not np.allclose(
            gr_betas, stage1_betas, rtol=1e-8, atol=1e-10
        ):
            raise RuntimeError("Stage1 HazeGen and Stage2 DiffDehaze schedules differ.")

        with torch.no_grad():
            self.degradation_text_condition = self.gr_model.clip.encode(
                [self.cfg.degradation_prompt]
            ).detach()

        print("[HazeCDG] Proposed algorithm : Haze Counter-Degradation Guidance")
        print("[HazeCDG] sensing signal     : g_t = eps_D - eps_base")
        print("[HazeCDG] guidance rule      : eps_R^HazeCDG = eps_R - 2 g_t")
        print("[HazeCDG] geometric meaning  : +g_t -> -g_t around eps_base")
        print("[HazeCDG] sparse reuse       : latest g is reused causally between anchors")
        print(f"[HazeCDG] sensing K          : {self.cfg.anchor_counts}")
        print("[HazeCDG] non-anchor extras  : NO G_D / NO eps_base")
        print("[HazeCDG] tuned strength     : none")

    def _build_sampler(self) -> None:
        if self.cfg.mode == "baseline":
            self.sampler = self.OfficialSpacedSampler(
                self.gr_diffusion.betas,
                self.gr_diffusion.parameterization,
                rescale_cfg=False,
            )
            print("[sampler] baseline : official standard SpacedSampler.sample()")
            return

        if self.stage1_controlnet is None or self.degradation_text_condition is None:
            raise RuntimeError("HazeCDG mode did not initialize official Stage1 HazeGen.")

        SamplerClass = make_sparse_guidance_sampler_class(self.OfficialSpacedSampler)
        self.sampler = SamplerClass(
            self.gr_diffusion.betas,
            self.gr_diffusion.parameterization,
            rescale_cfg=False,
            stage1_controlnet=self.stage1_controlnet,
            degradation_text_condition=self.degradation_text_condition,
            anchor_count=self.cfg.anchor_counts[0],
            log_every=self.cfg.log_every,
        )
        print("[sampler] HazeCDG : standard SpacedSampler + counter-degradation guidance + causal sparse reuse")

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()

    @staticmethod
    def _scaled_retry_work_size(orig_h: int, orig_w: int, scale: float) -> Tuple[int, int]:
        scale = float(scale)
        if not (0.0 < scale < 1.0):
            raise ValueError(f"OOM fallback scale must be in (0,1), got {scale}.")
        h = max(64, int(round(int(orig_h) * scale)))
        w = max(64, int(round(int(orig_w) * scale)))
        return nearest_multiple_work_size(h, w, multiple=64)

    def _read_and_prepare(
        self,
        image_path: Path,
        *,
        fallback_scale: Optional[float] = None,
    ):
        """Preserve preprocessing exactly; never zero-pad."""
        image = self.cv2.imread(str(image_path), self.cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cv2 failed to read image: {image_path}")
        image = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
        image = self.ToTensor()(image).unsqueeze(0)
        _, _, orig_h, orig_w = image.shape

        if fallback_scale is None:
            if orig_h < 512 or orig_w < 512:
                image = self.rescaler(image)
            _, _, scaled_h, scaled_w = image.shape
            h_work, w_work = nearest_multiple_work_size(
                scaled_h, scaled_w, multiple=64
            )
            resize_reason = "preserved short-side512-if-needed + nearest64/no-pad"
        else:
            h_work, w_work = self._scaled_retry_work_size(
                orig_h, orig_w, fallback_scale
            )
            resize_reason = f"OOM nearest64 x{float(fallback_scale):.3f}"

        current_h, current_w = int(image.shape[-2]), int(image.shape[-1])
        if (h_work, w_work) != (current_h, current_w):
            image = F.interpolate(
                image,
                size=(h_work, w_work),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        if (h_work, w_work) != (orig_h, orig_w):
            print(
                f"[resize] {orig_w}x{orig_h} -> {w_work}x{h_work} "
                f"({resize_reason})"
            )
        if h_work % 64 != 0 or w_work % 64 != 0:
            raise RuntimeError(f"Preprocessing produced invalid size: {w_work}x{h_work}")
        image = image.to(self.device)
        return image, int(orig_h), int(orig_w), int(h_work), int(w_work)

    @torch.no_grad()
    def _run_one(
        self,
        image_path: Path,
        image_seed: int,
        *,
        anchor_count: Optional[int] = None,
        fallback_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        seed_all(image_seed)
        image, orig_h, orig_w, h_work, w_work = self._read_and_prepare(
            image_path, fallback_scale=fallback_scale
        )
        cond = self.gr_model.prepare_condition(image, [self.cfg.dehaze_prompt])

        if self.cfg.mode == "hazecdg":
            if anchor_count is None:
                raise ValueError("HazeCDG inference requires anchor_count.")
            self.sampler.set_anchor_count(int(anchor_count))
            self.sampler.set_degradation_condition_hint(cond["c_img"])

        z = self.sampler.sample(
            model=self.gr_model,
            device=str(self.device),
            steps=self.cfg.steps,
            x_size=cond["c_img"].shape,
            cond=cond,
            uncond=None,
            cfg_scale=1.0,
            progress=self.cfg.progress,
        )

        if self.cfg.mode == "hazecdg":
            qn = float(max(1, self.sampler.query_count))
            gn = float(max(1, self.sampler.guided_count))
            print(
                "[HazeCDG] summary"
                f" | K={anchor_count}/{self.cfg.steps}"
                f" | steps={self.sampler.call_index}"
                f" | G_D/base queries={self.sampler.query_count}"
                f" | reused steps={self.sampler.reused_step_count}"
                f" | anchor mean||g||={self.sampler.sum_anchor_g_norm/qn:.4e}"
                f" | anchor mean||corr||={self.sampler.sum_anchor_correction_norm/qn:.4e}"
                f" | all-step mean||corr||={self.sampler.sum_applied_correction_norm/gn:.4e}"
                f" | guidance factor=2.0"
            )

        result = (self.gr_model.vae_decode(z) + 1.0) / 2.0
        result = result[:, :, :h_work, :w_work].clip(0.0, 1.0)
        result = result.detach().cpu()
        del z, image, cond

        if self.cfg.save_original_size and (h_work, w_work) != (orig_h, orig_w):
            result = self.Resize(
                (orig_h, orig_w),
                interpolation=self.InterpolationMode.BICUBIC,
                antialias=True,
            )(result)

        meta = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "work_h": h_work,
            "work_w": w_work,
        }
        return result, meta

    def _output_path_for_root(self, root: Path, image_path: Path) -> Path:
        try:
            rel = image_path.relative_to(self.cfg.input_root)
        except ValueError:
            rel = Path(image_path.name)
        return (root / rel).with_suffix(".png")

    def _save_tensor_atomic(self, result: torch.Tensor, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(out_path.stem + ".tmp.png")
        self.torchvision.utils.save_image(result.squeeze(0), str(tmp_path))
        os.replace(tmp_path, out_path)
        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise IOError(f"Output verification failed: {out_path}")

    def _clear_cuda_after_oom(self) -> None:
        if hasattr(self, "sampler"):
            if hasattr(self.sampler, "reset_image_state"):
                try:
                    self.sampler.reset_image_state()
                except Exception:
                    pass
            if hasattr(self.sampler, "degradation_condition_hint"):
                self.sampler.degradation_condition_hint = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _run_baseline(self, files: Sequence[Path]) -> None:
        root = self.cfg.output_root
        root.mkdir(parents=True, exist_ok=True)
        total = len(files)
        for idx, image_path in enumerate(files, 1):
            relative_name = image_path.relative_to(self.cfg.input_root).as_posix()
            out_path = self._output_path_for_root(root, image_path)
            if (
                self.cfg.skip_existing
                and out_path.is_file()
                and out_path.stat().st_size > 0
            ):
                print(f"[{idx:04d}/{total:04d}] [skip] {relative_name} -> {out_path}")
                continue
            print(f"\n[{idx:04d}/{total:04d}] {relative_name} | seed={self.cfg.seed}")
            attempts: List[Optional[float]] = [None]
            if self.cfg.oom_retry_enabled:
                attempts += list(self.cfg.oom_retry_scales)
            success = False
            last_oom: Optional[Exception] = None
            for fallback_scale in attempts:
                try:
                    result, meta = self._run_one(
                        image_path,
                        self.cfg.seed,
                        fallback_scale=fallback_scale,
                    )
                    self._save_tensor_atomic(result, out_path)
                    del result
                    reason = (
                        "controlled-preprocess" if fallback_scale is None
                        else f"oom-nearest64-x{fallback_scale:.3f}"
                    )
                    print(
                        f"[save] {out_path} | work={meta['work_w']}x{meta['work_h']} "
                        f"| reason={reason}"
                    )
                    success = True
                    break
                except Exception as exc:
                    if not (self.cfg.oom_retry_enabled and self._is_cuda_oom(exc)):
                        raise
                    last_oom = exc
                    self._clear_cuda_after_oom()
                    print(f"[OOM] baseline | fallback={fallback_scale}")
            if not success:
                raise RuntimeError(f"CUDA OOM persisted for {relative_name}.") from last_oom

    def _run_hazecdg_sweep(self, files: Sequence[Path]) -> None:
        ks = sorted(self.cfg.anchor_counts, reverse=True)
        roots = {k: self.cfg.output_root / f"K{k:03d}" for k in ks}
        for root in roots.values():
            root.mkdir(parents=True, exist_ok=True)

        total = len(files)
        for idx, image_path in enumerate(files, 1):
            relative_name = image_path.relative_to(self.cfg.input_root).as_posix()
            final_paths = {
                k: self._output_path_for_root(roots[k], image_path) for k in ks
            }
            if self.cfg.skip_existing and all(
                p.is_file() and p.stat().st_size > 0 for p in final_paths.values()
            ):
                print(f"[{idx:04d}/{total:04d}] [skip all K] {relative_name}")
                continue

            active_ks = (
                list(ks)
                if self.cfg.oom_retry_enabled
                else [
                    k for k in ks
                    if not (
                        self.cfg.skip_existing
                        and final_paths[k].is_file()
                        and final_paths[k].stat().st_size > 0
                    )
                ]
            )
            print(
                f"\n[{idx:04d}/{total:04d}] {relative_name} "
                f"| seed={self.cfg.seed} | pending K={active_ks}"
            )

            attempts: List[Optional[float]] = [None]
            if self.cfg.oom_retry_enabled:
                attempts += list(self.cfg.oom_retry_scales)
            success = False
            last_oom: Optional[Exception] = None
            for fallback_scale in attempts:
                temp_paths: List[Path] = []
                first_meta: Optional[Dict[str, Any]] = None
                try:
                    for k in active_ks:
                        label = (
                            "controlled-preprocess" if fallback_scale is None
                            else f"OOM nearest64 x{fallback_scale:.3f}"
                        )
                        print(f"[HazeCDG] run K={k} | {label}")
                        result, meta = self._run_one(
                            image_path,
                            self.cfg.seed,
                            anchor_count=k,
                            fallback_scale=fallback_scale,
                        )
                        if first_meta is None:
                            first_meta = dict(meta)
                        elif (
                            meta["work_h"] != first_meta["work_h"]
                            or meta["work_w"] != first_meta["work_w"]
                        ):
                            raise RuntimeError(
                                "HazeCDG K variants used different working resolutions."
                            )
                        out_path = final_paths[k]
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        tmp_path = out_path.with_name(out_path.stem + ".pending.png")
                        self.torchvision.utils.save_image(result.squeeze(0), str(tmp_path))
                        temp_paths.append(tmp_path)
                        del result

                    for k in active_ks:
                        out_path = final_paths[k]
                        tmp_path = out_path.with_name(out_path.stem + ".pending.png")
                        os.replace(tmp_path, out_path)
                        print(f"[save] K={k} -> {out_path}")
                    success = True
                    break
                except Exception as exc:
                    for tmp_path in temp_paths:
                        try:
                            tmp_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                    if not (self.cfg.oom_retry_enabled and self._is_cuda_oom(exc)):
                        raise
                    last_oom = exc
                    self._clear_cuda_after_oom()
                    print(f"[OOM] {relative_name} | fallback={fallback_scale}")
            if not success:
                raise RuntimeError(f"CUDA OOM persisted for {relative_name}.") from last_oom

    def run(self) -> None:
        files = list_images(self.cfg.input_root)
        if not files:
            raise RuntimeError(f"No input images found under: {self.cfg.input_root}")
        if self.cfg.max_images > 0:
            files = files[: self.cfg.max_images]

        self.cfg.output_root.mkdir(parents=True, exist_ok=True)
        print("=" * 108)
        print("HazeCDG :: counter-degradation guidance + sparse causal degradation reuse")
        print("=" * 108)
        print(f"Mode         : {self.cfg.mode}")
        print(f"Dataset      : {DATASET_OUTPUT_NAMES[self.cfg.dataset]}")
        print(f"Input root   : {self.cfg.input_root}")
        print(f"Images       : {len(files)}")
        print(f"Output root  : {self.cfg.output_root}")
        print("AccSamp      : OFF")
        print(f"Steps        : {self.cfg.steps}")
        print(f"Seed policy  : constant seed={self.cfg.seed} reset for every image / K")
        print("Preprocess   : short-side512-if-needed + nearest64; no pad")
        print(f"G_R weight   : {self.cfg.stage2_checkpoint}")
        if self.cfg.mode == "hazecdg":
            print(f"G_D weight   : {self.cfg.stage1_checkpoint}")
            print(f"Sensing K    : {sorted(self.cfg.anchor_counts)}")
            for k in sorted(self.cfg.anchor_counts):
                print(
                    f"  K={k:<3d} ({100.0*k/self.cfg.steps:5.1f}%) "
                    f"-> {self.cfg.output_root / f'K{k:03d}'}"
                )
        print("=" * 108)

        if self.cfg.mode == "baseline":
            self._run_baseline(files)
        else:
            self._run_hazecdg_sweep(files)
        print("\n[done] LHD baseline/HazeCDG inference completed.")


# =============================================================================
# Config construction
# =============================================================================


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def build_runtime_config(
    project_root: Path,
    config_path: Path,
    cfg_dict: Dict[str, Any],
) -> RuntimeConfig:
    mode = normalize_mode(nested_get(cfg_dict, ["experiment", "mode"], "hazecdg"))

    repo_root = resolve_path(
        project_root,
        nested_get(
            cfg_dict, ["paths", "repo_root"],
            "./third_party/Learning-Hazing-to-Dehazing",
        ),
    )
    if not repo_root.is_dir():
        raise FileNotFoundError(f"paths.repo_root does not exist: {repo_root}")

    stage1_config = resolve_path(
        project_root,
        nested_get(
            cfg_dict, ["paths", "stage1_config"],
            "./third_party/Learning-Hazing-to-Dehazing/configs/inference/stage1.yaml",
        ),
    )
    stage2_config = resolve_path(
        project_root,
        nested_get(
            cfg_dict, ["paths", "stage2_config"],
            "./third_party/Learning-Hazing-to-Dehazing/configs/inference/stage2.yaml",
        ),
    )
    sd_checkpoint = resolve_path(
        project_root,
        nested_get(
            cfg_dict, ["paths", "sd_checkpoint"],
            "./checkpoints/Learning-Hazing-to-Dehazing/v2-1_512-ema-pruned.ckpt",
        ),
    )
    stage1_checkpoint = resolve_path(
        project_root,
        nested_get(
            cfg_dict, ["paths", "stage1_checkpoint"],
            "./checkpoints/Learning-Hazing-to-Dehazing/stage1.pt",
        ),
    )
    stage2_checkpoint = resolve_path(
        project_root,
        nested_get(
            cfg_dict, ["paths", "stage2_checkpoint"],
            "./checkpoints/Learning-Hazing-to-Dehazing/stage2.pt",
        ),
    )

    _require_file(stage2_config, "Official Stage2 config")
    _require_file(sd_checkpoint, "SD2.1 checkpoint")
    _require_file(stage2_checkpoint, "DiffDehaze Stage2 checkpoint")
    if mode == "hazecdg":
        _require_file(stage1_config, "Official Stage1 config")
        _require_file(stage1_checkpoint, "HazeGen Stage1 checkpoint")

    dataset = normalize_dataset_name(nested_get(cfg_dict, ["data", "dataset"], "RTTS"))
    dataset_root = resolve_path(
        project_root,
        nested_get(cfg_dict, ["data", "dataset_root"], "./dataset"),
    )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"data.dataset_root does not exist: {dataset_root}")
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

    steps = int(nested_get(cfg_dict, ["inference", "steps"], 50))
    if steps <= 0 or steps > 1000:
        raise ValueError("inference.steps must be in [1, 1000].")

    raw_anchor_counts = nested_get(
        cfg_dict, ["hazecdg", "anchor_counts"], [5]
    )
    if isinstance(raw_anchor_counts, (int, float)):
        raw_anchor_counts = [int(raw_anchor_counts)]
    anchor_counts = sorted({int(x) for x in raw_anchor_counts})
    if not anchor_counts or any(k <= 0 for k in anchor_counts):
        raise ValueError("hazecdg.anchor_counts must contain positive integers.")
    if any(k > steps for k in anchor_counts):
        raise ValueError(
            "Every hazecdg.anchor_counts value must be <= inference.steps "
            f"(steps={steps}, K={anchor_counts})."
        )

    log_every = int(nested_get(cfg_dict, ["hazecdg", "log_every"], 10))
    if log_every < 0:
        raise ValueError("hazecdg.log_every must be >= 0.")

    oom_retry_scales = normalize_oom_retry_scales(
        nested_get(
            cfg_dict, ["runtime", "oom_retry_scales"],
            [0.75, 0.50, 0.375, 0.25],
        )
    )

    return RuntimeConfig(
        project_root=project_root,
        repo_root=repo_root,
        stage1_config=stage1_config,
        stage2_config=stage2_config,
        sd_checkpoint=sd_checkpoint,
        stage1_checkpoint=stage1_checkpoint,
        stage2_checkpoint=stage2_checkpoint,
        dataset=dataset,
        dataset_root=dataset_root,
        input_root=input_root,
        output_root=output_root,
        mode=mode,
        device=str(nested_get(cfg_dict, ["runtime", "device"], "cuda:0")),
        expected_conda_env=str(
            nested_get(
                cfg_dict, ["runtime", "expected_conda_env"],
                "LHD_xformers_blackwell",
            )
        ),
        attention_mode=str(
            nested_get(cfg_dict, ["runtime", "attention_mode"], "xformers")
        ).strip().lower(),
        cuda_visible_devices=str(
            nested_get(cfg_dict, ["runtime", "cuda_visible_devices"], "") or ""
        ),
        seed=int(nested_get(cfg_dict, ["runtime", "seed"], 231)),
        max_images=max_images,
        skip_existing=bool(nested_get(cfg_dict, ["data", "skip_existing"], True)),
        steps=steps,
        dehaze_prompt=str(
            nested_get(cfg_dict, ["inference", "dehaze_prompt"], "remove dense fog")
        ),
        degradation_prompt=str(
            nested_get(
                cfg_dict, ["hazecdg", "degradation_prompt"],
                "hazy, foggy, misty, obscure, smoggy.",
            )
        ),
        anchor_counts=anchor_counts,
        progress=bool(nested_get(cfg_dict, ["inference", "progress"], True)),
        log_every=log_every,
        save_original_size=bool(
            nested_get(cfg_dict, ["runtime", "save_original_size"], True)
        ),
        oom_retry_enabled=bool(
            nested_get(cfg_dict, ["runtime", "oom_retry_enabled"], False)
        ),
        oom_retry_scales=oom_retry_scales,
        raw_cfg=cfg_dict,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LHD baseline / HazeCDG counter-degradation guidance "
            "with sparse causal degradation reuse."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./configs/eval_LHD_HazeCDG.yaml",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["RTTS", "URHI", "Fattal"],
        default="",
        help="Override data.dataset from YAML.",
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

    # CLI dataset overrides YAML.
    if args.dataset:
        cfg_dict.setdefault("data", {})
        cfg_dict["data"]["dataset"] = args.dataset

    runtime_cfg = build_runtime_config(root, config_path, cfg_dict)
    runner = LHDRunner(runtime_cfg)

    if args.check_only:
        files = list_images(runtime_cfg.input_root)
        if runtime_cfg.max_images > 0:
            files = files[: runtime_cfg.max_images]

        print("=" * 100)
        print("[check] LHD/HazeCDG configuration loaded.")
        print(f"[check] mode         : {runtime_cfg.mode}")
        print(f"[check] dataset      : {DATASET_OUTPUT_NAMES[runtime_cfg.dataset]}")
        print(f"[check] input images : {len(files)}")
        print(f"[check] steps        : {runtime_cfg.steps}")
        print("[check] AccSamp      : OFF")
        print("[check] HazeCDG rule  : eps_R - 2 g")
        print("[check] geometry      : eps_D=eps_base+g, eps_C=eps_base-g")

        if runtime_cfg.mode == "hazecdg":
            print(f"[check] anchor counts: {runtime_cfg.anchor_counts}")
            for k in runtime_cfg.anchor_counts:
                anchors = choose_anchor_eval_indices(
                    steps=runtime_cfg.steps,
                    anchor_count=k,
                )
                print(f"[check] K={k:<3d} calls : {anchors}")

            print(
                "[check] non-anchor   : "
                "latest sensed haze differential is reused causally"
            )
            print("[check] K=steps      : dense per-step HazeCDG guidance")

        print("[check] Inference was not started.")
        print("=" * 100)
        return

    runner.run()


if __name__ == "__main__":
    main()