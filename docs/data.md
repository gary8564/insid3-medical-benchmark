# Data preparation

Instructions to download and cache the three **in-context segmentation** domains used in this benchmark. Tests never download data (synthetic fixtures only).

Run all commands from the **repository root**. After preparation, the layout is:

```text
insid3-medical-benchmark/
├── data/
│   ├── raw/
│   │   ├── kvasir-seg/          # Kvasir-SEG zip + extracted images/masks
│   │   ├── kipa22/              # KiPA22 train.zip (image + label NIfTI)
│   │   └── acdc/                # ACDC training ED/ES frames + gt (no 4D, no test)
│   ├── processed/
│   │   ├── polyp/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   ├── kidney_tumor/
│   │   │   ├── images/
│   │   │   ├── masks/
│   │   │   └── slice_stats.json
│   │   └── cardiac/
│   │       ├── images/
│   │       └── masks/
├── pretrain/                    # DINOv3 checkpoints (gated; gitignored)
└── third_party/INSID3/          # git submodule (visinf/INSID3, pinned commit)
```

`data/` and `pretrain/` are gitignored. Colab should read `data/processed/` and `pretrain/` from Drive rather than re-downloading 3D volumes.

```bash
git clone --recurse-submodules <this-repo-url>
# or, if already cloned:
git submodule update --init --recursive
```

Only **splits that ship segmentation masks** are downloaded. Challenge test sets without public GT are skipped.

```bash
uv run python src/data/prepare.py
uv run python src/data/prepare.py --datasets polyp          # one domain
uv run python src/data/prepare.py --skip-download           # re-slice existing data/raw/
```

---

## 🩺 Kvasir-SEG (colon polyp)

Official source: [Simula Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) (~1,000 pairs, 46 MB). Paper: Jha et al., MMM 2020.

**Masks:** binary 1-bit images (white = polyp, black = background). Not multi-class. The older Kvasir set is GI *disease classification*; this zip is the polyp subset with paired masks.

**Download:** the full Simula zip (`images/` + `masks/`). Every image has a mask.

```bash
uv run python src/data/download_kvasir.py
# or: uv run python src/data/prepare.py --datasets polyp
```

This should result in **1,000** processed pairs under `data/processed/polyp/`. Eval samples 600 random 1-shot episodes from that folder (`--seed 0`).

---

## 🫘 KiPA22 (kidney tumor)

Official challenge: [KiPA22 Grand Challenge](https://kipa22.grand-challenge.org/). Training mirror: [YongchengYAO/KiPA22](https://huggingface.co/datasets/YongchengYAO/KiPA22) (`train.zip`, 407 MB). Cite He et al., MedIA 2021.

**What we download:** **training only** (`train/image/{0..69}.nii.gz` + `train/label/{0..69}.nii.gz`). The 30 open-test and 30 closed-test cases are **not** fetched (no public GT on the regular challenge test sets).

**Voxel labels** ([official eval](https://github.com/KiPA2022/kipa22/blob/main/EVALUATION/evaluation.py)):

| Value | Structure |
|---|---|
| 0 | background |
| 1 | renal vein |
| 2 | kidney |
| 3 | renal artery |
| 4 | kidney tumor |

Headline target is **label 4 (tumor)** only. One max-tumor-area axial PNG per case. Tumor pixel areas are recorded in `data/processed/kidney_tumor/slice_stats.json`.

```bash
uv run python src/data/download_kipa.py
# or: uv run python src/data/prepare.py --datasets kidney_tumor
```

This should result in **70** processed 2D pairs (one per training case).

---

## ❤️ ACDC (cardiac LV cavity)

Official challenge: [CREATIS ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/). Unmodified NIfTI mirror: [MedOtter/ACDC](https://huggingface.co/datasets/MedOtter/ACDC). Cite Bernard et al., TMI 2018.

**What we download:** **training patients 001–100** only (`Info.cfg` + `patientXXX_frameYY.nii.gz` + `_gt.nii.gz`). We skip:

- `testing/` (challenge test GT is not in the public training pack)
- `*_4d.nii.gz` cine volumes
- intensity-normalized / scribble / `.h5` reprocesses (e.g. WSL4MIS / Kaggle packs)

**Voxel labels** ([ACDC evaluation](https://www.creatis.insa-lyon.fr/Challenge/acdc/evaluation.html)):

| Value | Structure |
|---|---|
| 0 | background |
| 1 | RV cavity |
| 2 | myocardium |
| 3 | LV cavity |

Headline target is **label 3 (LV cavity)** on the **ED** frame (`Info.cfg` → `ED:`), one max-LV-area short-axis PNG per patient. Papillary muscles are inside the cavity in the official protocol.

```bash
uv run python src/data/download_acdc.py
# or: uv run python src/data/prepare.py --datasets cardiac
```

This should result in **100** processed 2D pairs (training patients).

Native intensities are stretched **per slice** to 8-bit (1–99 percentile) only for the PNG cache INSID3/PIL can load. That stretch is not applied to the original NIfTI.

---

## 🧱 DINOv3 weights

INSID3 needs a **frozen DINOv3** checkpoint. Access is gated by Meta ([facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)). Request access now; approval can lag.

```text
pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth   # headline (Large)
pretrain/dinov3_vits16_pretrain_lvd1689m-08c60483.pth   # local check (Small)
pretrain/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth   # optional
pretrain/dinov2_vitl14_pretrain.pth                    # Matcher + GF-SAM (public)
pretrain/sam_vit_h_4b8939.pth                          # Matcher + GF-SAM (public, ~2.4 GB)
pretrain/flexict_2d_teacher.pth                        # Phase 7, CC BY-NC-ND 4.0
pretrain/MedSAM2_latest.pt                             # Phase 8
```

Public checkpoints (DINOv2, SAM ViT-H, MedSAM2):

```bash
uv run python src/data/download_weights.py --all-public
```

FlexiCT-2D is a Google Drive file — save it as `pretrain/flexict_2d_teacher.pth`.

---

## 📦 INSID3 checkout

INSID3 is a git submodule at `third_party/INSID3` (not a loose clone). Init it before inference:

```bash
git submodule update --init --recursive
```

---

## 📍 Runs

Headline eval follows INSID3 [`datasets/lung.py`](https://github.com/visinf/INSID3/blob/main/datasets/lung.py): **600** independently sampled 1-shot episodes (`--seed 0`) from every paired image under `data/processed/<domain>/`. Each episode draws a random target and a different random reference. `--dry-run` lists the pairs; `--preview` randomly picks N of them (same seed).

```bash
# List 600 random 1-shot episodes (no DINOv3)
uv run python src/run_insid3.py --dataset polyp --dry-run

# First-pass: randomly pick N of the 600 sampled episodes
# N defaults to PREVIEW_N in src/data/constants.py (currently 8)
uv run python src/run_insid3.py --dataset polyp --preview
uv run python src/run_insid3.py --dataset polyp --preview 3

# Cheap local check (ViT-S, one episode, CPU). Not a result.
uv run python src/run_insid3.py --dataset polyp --model-size small --preview 1 --device cpu

# Headline: 600 random 1-shot episodes, ViT-L, --image-size 768, --seed 0, no CRF
uv run python src/run_insid3.py --dataset polyp
uv run python src/run_insid3.py --dataset kidney_tumor
uv run python src/run_insid3.py --dataset cardiac

# Phase 6–8 reuse results/<dataset>/episodes.json (never re-sample)
python -m src.methods.gfsam --dataset polyp --episodes-json results/polyp/episodes.json
python -m src.methods.matcher --dataset polyp --episodes-json results/polyp/episodes.json
python -m src.methods.insid3 --backbone flexict2d --dataset kidney_tumor \
  --episodes-json results/kidney_tumor/episodes.json \
  --output-dir results/flexict2d/debiased
python -m src.methods.medsam2 --dataset polyp --episodes-json results/polyp/episodes.json
```

INSID3 JSON and predicted masks go under `--output-dir` (`episodes.json` or `metrics.json`, plus `preds/`). The default is `results/<dataset>/`. Ablations pass a nested run directory, e.g. `results/cardiac/five-shot`. Phase 6–8 write `results/<method>/<dataset>/` and do not overwrite the Phase 3 snapshot.

If `data/processed` exists:

```bash
uv run pytest -m slow
```
