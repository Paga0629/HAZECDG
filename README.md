# HAZECDG: Reusing Learned Degradation Models via Counter-Degradation Guidance for Diffusion Dehazing

This repository provides the HAZECDG evaluation pipeline with wrappers for **Learning-Hazing-to-Dehazing (LHD / DiffDehaze)** and **DOD**.

---

## ⚙️ 1. Environment Setup

> Environment is prepared for NVIDIA Blackwell GPUs such as the **NVIDIA RTX PRO 6000 Blackwell Workstation Edition**.  
> PyTorch uses CUDA 12.8, and xFormers is built from source for Blackwell `SM120`.
> Nvidia Driver Version: 580.173.02.

### 1.1 Conda environment

```bash
conda create -n hazecdg --file explicit.txt -y
conda activate hazecdg
```

### 1.2 Pip Packages

```bash
python -m pip install \
  --no-deps \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements-pip.txt
```

### 1.3 Setuptools / Wheel

```bash
python -m pip install --no-deps --force-reinstall \
  setuptools==80.9.0 \
  wheel==0.48.0
```

### 1.4 OpenCV

```bash
python -m pip install --no-deps --force-reinstall \
  opencv-python-headless==5.0.0.93
```

### 1.5 Xformer Capability

```bash
python - <<'PY'
from pathlib import Path
import xformers

p = Path(xformers.__file__).parent / "ops" / "fmha" / "cutlass.py"

old = "CUDA_MAXIMUM_COMPUTE_CAPABILITY = (9, 0)"
new = "CUDA_MAXIMUM_COMPUTE_CAPABILITY = (12, 0)"

s = p.read_text()

if new in s:
    print("[OK] xformers already patched")
elif old in s:
    p.write_text(s.replace(old, new, 1))
    print("[OK] xformers Blackwell patch applied")
else:
    raise RuntimeError(f"Unexpected xformers file: {p}")
PY
```

## 📦 2. Third-Party Baselines

Run the following commands from the HAZECDG project root:

```bash
mkdir -p datasets
mkdir -p third_party
mkdir -p checkpoints/Learning-Hazing-to-Dehazing
mkdir -p checkpoints/DOD

git clone https://github.com/ruiyi-w/Learning-Hazing-to-Dehazing.git \
    third_party/Learning-Hazing-to-Dehazing

git clone https://github.com/tonia86/DOD.git \
    third_party/DOD
```

---

## 📥 3. Pretrained Checkpoints

Pretrained weights will be provided separately.

| Checkpoint package | Download |
|---|---|
| LHD / DiffDehaze | [Download](https://github.com/ruiyi-w/Learning-Hazing-to-Dehazing) |
| DOD | [Download](https://github.com/tonia86/DOD?utm_source=chatgpt.com) |

Place the downloaded files under `./checkpoints/`:

```text
checkpoints/
├── Learning-Hazing-to-Dehazing/
│   ├── v2-1_512-ema-pruned.ckpt
│   ├── stage1.pt
│   └── stage2.pt
└── DOD/
    ├── sd21
    ├── stage1.pkl
    ├── stage2.pkl
    └── mfm/
        └── 256x256_diffusion_uncond.pt
```

---

## 🗂️ 4. Dataset Preparation

Place the evaluation datasets under `./datasets/`:

```text
datasets/
├── RTTS/
├── URHI/
└── Fattal/
```

---

## 🚀 5. Run Experiments

### LHD / DiffDehaze

```bash
python scripts/eval_LHD_HazeCDG.py \
    --config configs/eval_LHD_HazeCDG.yaml \

python scripts/eval_metrics.py \
    --config configs/eval_LHD_HazeCDG.yaml \
```

### DOD

```bash
python scripts/eval_DOD_HazeCDG.py \
    --config configs/eval_DOD_HazeCDG.yaml \

python scripts/eval_metrics.py \
    --config configs/eval_DOD_HazeCDG.yaml \
```

Dataset and output paths can be configured in:

```text
configs/eval_LHD_HazeCDG.yaml
configs/eval_DOD_HazeCDG.yaml
```
---

## Notes

- xFormers `0.0.33.post2` is built from source for Blackwell `SM120`.
- Do not install the original LHD or DOD requirements on top of the provided environment.
- Keep the YAML configuration used for each experiment together with the generated outputs and logs for reproducibility.
