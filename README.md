# INSID3 Cross-Domain Medical Segmentation Benchmark

This repository is a **cross-domain medical segmentation stress test** of [INSID3](https://github.com/visinf/INSID3) ([Cuttano et al., CVPR 2026 Oral](https://visinf.github.io/INSID3/)): training-free in-context segmentation on a frozen [DINOv3](https://github.com/facebookresearch/dinov3) backbone.

INSID3 has already been shown to work on medical images (ISIC skin lesions 54.4% mIoU, chest X-ray lung fields 78.8% mIoU). This project asks a narrower question: **does that easy/hard gap replicate on three new domains it has never been evaluated on?** The design crosses **imaging modality** (endoscopy / CT / MRI) with **target difficulty** (diffuse, low-contrast pathology vs. anatomically well-defined structure). 

| | Diffuse / low-contrast target | Clear anatomical boundary |
|---|---|---|
| **Endoscopy** | Colon polyp (Kvasir-SEG) | — |
| **CT** | Kidney tumor (KiPA22) | — |
| **MRI** | — | Cardiac LV cavity (ACDC) |

Evaluation is one-shot INSID3 only (**600 random 1-shot episodes** per domain: random target + a different random reference, seed 0, `--image-size 768`). Predicted masks are bilinearly upsampled from the patch grid.
This study does **not** apply INSID3’s optional CRF refinement. 

---

## Research Question

<!-- Does INSID3's ISIC (~54% mIoU) vs CXR (~79% mIoU) pattern hold on polyp / kidney tumor / cardiac LV under the same untuned pipeline? -->

## Datasets

<!-- Kvasir-SEG (46 MB, 2D). KiPA22 (~407 MB), one max-tumor axial slice per case. ACDC unmodified NIfTI (CREATIS or MedOtter; not WSL4MIS/Kaggle reprocesses), one ED LV slice per patient. -->

## Methodology

<!-- Local: uv sync, pytest on synthetic fixtures, download/preprocess, one-pair INSID3 check. Colab: same CLI on the 2D cache. Untuned τ=0.6, α=0.2, s=500. -->

## Installation

Phase 0 (laptop, no GPU, no dataset download):

```bash
uv sync --group dev
uv run pytest
```

Python 3.10 is pinned in `.python-version` (INSID3). `uv` will install it if needed.

Phase 2+ (local INSID3 / Colab):

```bash
git submodule update --init --recursive
uv sync --extra torch --group dev
```

Phase 6–8 (Matcher / GF-SAM / FlexiCT / MedSAM2) on a local GPU:

```bash
uv sync --extra torch --extra methods --group dev
```

If you cloned this repo without submodules, that `git submodule update --init --recursive` step is required. A new clone can use `git clone --recurse-submodules` instead. See [third_party/README.md](third_party/README.md).

Colab (torch already installed): do **not** `uv pip install .[torch]` (that would download a second torch). Install project deps plus the methods extra:

```bash
uv pip install --system nibabel einops huggingface_hub pycocotools
uv pip install --system opencv-python-headless pot omegaconf iopath timm scipy hydra-core
```

Request [DINOv3](https://github.com/facebookresearch/dinov3) weight access now. See [docs/data.md](docs/data.md).

## Usage

```bash
# Download + 2D cache (masked training images only)
uv run python src/data/prepare.py
uv run python src/data/prepare.py --datasets polyp
uv run python src/data/prepare.py --skip-download

# Sample 600 random 1-shot episodes (INSID3 lung.py); does not load DINOv3
uv run python src/run_insid3.py --dataset polyp --dry-run

# Random subset of N sampled episodes (N from --preview or src/data/constants.py PREVIEW_N)
uv run python src/run_insid3.py --dataset polyp --preview
uv run python src/run_insid3.py --dataset polyp --preview 3

# Cheap local check (ViT-S, one episode, CPU). Not a headline result.
uv run python src/run_insid3.py --dataset polyp --model-size small --preview 1 --device cpu
```

Headline eval (typically Colab GPU) omits `--preview` (ViT-L, `--image-size 768`, 600 random 1-shot episodes, `--seed 0`). Dataset details: [docs/data.md](docs/data.md).

Phase 6–8 reuse the Phase 3 `episodes.json` (same 600 pairs). Smoke with `--preview 1` before a full run:

```bash
python -m src.methods.gfsam --dataset polyp --preview 1 \
  --episodes-json results/polyp/episodes.json
python -m src.methods.matcher --dataset polyp --preview 1 \
  --episodes-json results/polyp/episodes.json
python -m src.methods.insid3 --backbone flexict2d --dataset kidney_tumor --preview 1 \
  --episodes-json results/kidney_tumor/episodes.json \
  --output-dir results/flexict2d/debiased
python -m src.methods.medsam2 --dataset polyp --preview 1 \
  --episodes-json results/polyp/episodes.json
```

## Results

<!-- INSID3-only three-domain table. Qualitative figures per domain. Do not merge Curia-2 numbers here. -->

## Backbone Ablation (Extension)

<!-- Stretch: Curia-2 encoder swap on CT/MRI only, debiasing on and off. Separate from the headline table. -->

## Known Risks

<!-- KiPA tumor size vs token resolution; cropping changes the task; ACDC LV-only; gated DINOv3 weights; reject intensity-normalized ACDC mirrors; one slice per patient. -->

## Citation

If you use this benchmark, please cite INSID3:

```bibtex
@inproceedings{cuttano2026insid3,
  title     = {{INSID3}: Training-Free In-Context Segmentation with {DINOv3}},
  author    = {Claudia Cuttano and Gabriele Trivigno and Christoph Reich and Daniel Cremers and Carlo Masone and Stefan Roth},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
