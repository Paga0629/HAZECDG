# HAZECDG: Reusing Learned Degradation Models via Counter-Degradation Guidance for Diffusion Dehazing

This repository provides the HAZECDG evaluation pipeline with wrappers for **Learning-Hazing-to-Dehazing (LHD / DiffDehaze)** and **DOD**.

---

## ⚙️ 1. Environment Setup

> This environment is prepared for NVIDIA Blackwell GPUs such as the **NVIDIA RTX PRO 6000 Blackwell Workstation Edition**.  
> PyTorch uses CUDA 12.8, and xFormers is built from source for Blackwell `SM120`.

### 1.1 Create the Conda environment

```bash
conda env create -f environment.yml
conda activate LHD_xformers_blackwell
```

### 1.2 Install Python dependencies

```bash
pip install -r requirements.txt
```

> `xformers` is intentionally excluded from `requirements.txt` because the standard CUDA 12.8 wheel may not include kernels compiled for Blackwell `SM120`.

### 1.3 Build xFormers for Blackwell

```bash
export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
export MAX_JOBS=4

git clone --recursive \
    --branch v0.0.33.post2 \
    --depth 1 \
    https://github.com/facebookresearch/xformers.git \
    /tmp/xformers-v0.0.33.post2

cd /tmp/xformers-v0.0.33.post2

pip install \
    --no-build-isolation \
    --no-deps \
    .
```

Return to the HAZECDG project root after installation.

```bash
cd /path/to/HAZECDG
```

> If xFormers compilation uses too much system memory, reduce `MAX_JOBS`, for example:
>
> ```bash
> export MAX_JOBS=2
> ```

---

## 📦 2. Third-Party Baselines

Run the following commands from the HAZECDG project root:

```bash
mkdir -p dataset
mkdir -p third_party
mkdir -p checkpoints/Learning-Hazing-to-Dehazing
mkdir -p checkpoints/DOD

git clone https://github.com/ruiyi-w/Learning-Hazing-to-Dehazing.git \
    third_party/Learning-Hazing-to-Dehazing

git clone https://github.com/tonia86/DOD.git \
    third_party/DOD
```

> **Do not install the original dependencies from the third-party repositories.**
>
> The provided HAZECDG environment contains the unified dependency stack for both LHD and DOD. Installing their original requirements may overwrite the PyTorch, CUDA, or other package versions used by HAZECDG.

---

## 📥 3. Pretrained Checkpoints

Pretrained weights will be provided separately.

| Checkpoint package | Download |
|---|---|
| LHD / DiffDehaze | [Download](CHECKPOINT_LINK_LHD) |
| DOD | [Download](CHECKPOINT_LINK_DOD) |

Place the downloaded files under `./checkpoints/`:

```text
checkpoints/
├── Learning-Hazing-to-Dehazing/
│   ├── v2-1_512-ema-pruned.ckpt
│   ├── stage1.pt
│   └── stage2.pt
└── DOD/
    ├── sd21/
    ├── stage1.pkl
    ├── stage2.pkl
    └── mfm/
        └── 256x256_diffusion_uncond.pt
```

> Replace the placeholder links above with the final checkpoint URLs before release.

---

## 🗂️ 4. Dataset Preparation

Place the evaluation datasets under `./dataset/`:

```text
dataset/
├── RTTS/
├── URHI/
└── Fattal/
```

Only the datasets required for your experiment need to be prepared.

Dataset and output paths can be configured in:

```text
configs/eval_LHD_HazeCDG.yaml
configs/eval_DOD_HazeCDG.yaml
```

---

## 🚀 5. Run Experiments

### LHD / DiffDehaze

```bash
python scripts/eval_LHD_HazeCDG.py \
    --config configs/eval_LHD_HazeCDG.yaml
```

### DOD

```bash
python scripts/eval_DOD_HazeCDG.py \
    --config configs/eval_DOD_HazeCDG.yaml
```

---

## Notes

- xFormers `0.0.33.post2` is built from source for Blackwell `SM120`.
- Do not install the original LHD or DOD requirements on top of the provided environment.
- Keep the YAML configuration used for each experiment together with the generated outputs and logs for reproducibility.
