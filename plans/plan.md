# INSID3 Cross-Domain Medical Segmentation Stress Test

## 1. Research Question

Does INSID3 — training-free in-context segmentation on frozen DINOv3 features —
generalize across medical domains that differ in **imaging modality** (endoscopy /
CT / MRI) and **target difficulty** (diffuse, low-contrast pathology vs. a clear
anatomical boundary)?

The paper reports ISIC skin lesion 54.4% mIoU vs. Chest X-ray lung field 78.8%
mIoU. This project tests whether that gap holds on three new domains, using
**INSID3 only** (untuned defaults). The headline result is the three-domain table
plus qualitative failures, not a method comparison.

## 2. Design

| | Diffuse / low-contrast target | Clear anatomical boundary |
|---|---|---|
| **Endoscopy** | Colon polyp (Kvasir-SEG) | — |
| **CT** | Kidney tumor (KiPA22) | — |
| **MRI** | — | Cardiac LV cavity (ACDC) |

Headline episodes: **600 random 1-shot pairs per domain** (INSID3
[`datasets/lung.py`](https://github.com/visinf/INSID3/blob/main/datasets/lung.py):
independent `np.random.choice` of target and a different reference from all
masked images, `num=600`, seed 0). Default `--image-size 768`, no CRF. Optional
KiPA-only rerun at 1024 if VRAM allows. `--preview N` randomly picks N of the 600 for a first-pass subset;
it is not the headline table.

## 3. Datasets

Download **direct** small sources. Do not pull the MedVision catalogue.

### 3.1 Kvasir-SEG (polyp)

- [Simula](https://datasets.simula.no/kvasir-seg/), ~1,000 pairs, **46 MB**
- Headline: 600 random 1-shot episodes from all paired images on disk
- `--preview N` / `--dry-run` to inspect sampled (ref, tgt) pairs

### 3.2 KiPA22 (kidney tumor)

- [YongchengYAO/KiPA22](https://huggingface.co/datasets/YongchengYAO/KiPA22), **407 MB**, 70 CT cases
- Use **label 4 (tumor)** only (1=vein, 2=kidney, 3=artery, 4=tumor)
- One axial slice per case: max tumor area → ~70 2D pairs

### 3.3 ACDC (cardiac LV)

- Official training NIfTI from [CREATIS](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
  (100 patients) or unmodified [MedOtter/ACDC](https://huggingface.co/datasets/MedOtter/ACDC)
  (~2.45 GB). Cite Bernard et al., TMI 2018
- Native intensities only (no percentile-clip / min–max / scribble / `.h5` packs)
- **LV cavity** only; one short-axis ED slice per patient (largest LV area)

### 3.4 Disk

| Asset | Where it lives | Why |
|---|---|---|
| Git repo + INSID3 submodule | GitHub (laptop clone is for tests only) | Colab clones this |
| Kvasir / KiPA / ACDC **raw** | Colab ephemeral disk, then **deleted** | Do not keep 3D volumes locally or on Drive |
| Processed 2D PNGs | Drive `insid3-medical-benchmark/processed/` | ≪ 1 GB; the only data eval needs |
| DINOv3 ViT-L | Drive `insid3-medical-benchmark/pretrain/` | Gated; copy once from the laptop |
| DINOv3 ViT-S | Laptop `pretrain/` (optional) | Local one-pair pytest only |
| `results/` | Drive `insid3-medical-benchmark/results/` | Colab disk is ephemeral |

Laptop keeps the git checkout, tests, and optionally ViT-S. It does **not** need `data/raw/` or the three-domain PNG cache.

## 4. Local tests, Colab data + GPU

Slice rules, loaders, metrics, and the CLI are implemented and tested on the laptop with **synthetic** fixtures (Phase 0–2). Colab is the worker for **real** download/preprocess and for headline GPU eval. The notebook (`notebooks/colab_eval.ipynb`) has no unique model code: it clones the repo and calls `prepare.py` / `run_insid3.py` with `--input-dir` / `--output-dir` / `--processed-root` pointing at Drive.

| Laptop | Colab Pro |
|---|---|
| `uv sync --group dev` + pytest | Clone + `%pip` project deps (runtime already has torch) |
| Optional ViT-S one-pair (`--model-size small --preview 1 --device cpu`) | `prepare.py` once → Drive `processed/`, delete raw |
| No KiPA/ACDC volumes required | 600 episodes × 3 domains, ViT-L, GPU |
| Push to GitHub before Colab | No pytest in the notebook |

## 5. Implementation Phases

### Phase 0 — Env and tests (laptop, no full datasets) — done when `uv run pytest` is green

- `uv` + `pyproject.toml` (Python 3.10 locally). Colab uses the runtime Python and installs deps from the notebook, not `uv pip install .` (`package = false`, CPython pin is 3.10).
- Pytest on **synthetic fixtures** (pairing, KiPA label 4, ACDC ED + LV, 600 random 1-shot, mIoU/Dice, `--dry-run`).
- Must pass with no GPU and no dataset download.

### Phase 1 — Push the worker, then build the 2D cache on Colab

Laptop:

1. Commit and push (include `.gitmodules` and the `third_party/INSID3` gitlink).
2. Copy ViT-L to Drive: `MyDrive/insid3-medical-benchmark/pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`.
3. You may delete local `data/` to free disk. Keep ViT-S only if you still want the local one-pair test.

Colab (notebook section 1, CPU is enough for this step):

1. Open `notebooks/colab_eval.ipynb` from the GitHub copy (File → Open notebook from GitHub) or upload it. GPU optional here.
2. Set `REPO_URL`. Mount Drive.
3. Clone `--recurse-submodules`. Symlink `pretrain/` to Drive.
4. Hugging Face login (KiPA + ACDC).
5. `python src/data/prepare.py --processed-root <Drive>/processed` (skips domains that already have PNGs).
6. Delete Colab `data/raw/`. Confirm counts: polyp ≥ 1000, kidney_tumor = 70, cardiac = 100.
7. Do not copy raw NIfTI to Drive.

### Phase 2 — Local INSID3 one-pair check — done when `test_insid3_one_pair` passes

- Laptop: `git submodule update --init`, ViT-S in `pretrain/`, `uv sync --extra torch --group dev`.
- One synthetic pair through INSID3. Wiring only — not a result.
- Optional CLI: `uv run python src/run_insid3.py --dataset polyp --model-size small --preview 1 --device cpu` (needs a local polyp cache).

### Phase 3 — Colab Pro headline eval

Notebook: `notebooks/colab_eval.ipynb`. Same flags as a local CLI; Drive paths only.

Protocol (untuned): τ=0.6, α=0.2, s=500, `--image-size 768`, `--seed 0`, `--model-size large`, **no** `--preview`. Metrics JSON is written by `run_insid3.py` (not a separate `evaluate.py` process).

Per domain (set `DATASET`, run the eval cell; one domain per session if the VM dies):

```text
python src/run_insid3.py \
  --dataset polyp \
  --input-dir <Drive>/processed \
  --output-dir <Drive>/results \
  --model-size large --image-size 768 --seed 0 --device cuda
```

Then `kidney_tumor`, then `cardiac`. If ViT-L OOM, use `--model-size base` and report which. Optional `--preview 8` is a first-pass only.

After each domain, confirm Drive has `results/<dataset>/metrics.json` and `preds/`.

### Phase 4 — KiPA resolution ablation

- Same 600 KiPA episodes at `--image-size 1024` only after Phase 3 used 768. Full-image, no crop. Write to a distinct output dir (e.g. `results/kidney_tumor_1024/`) so 768 is not overwritten.

### Phase 5 — Evaluate and write up

- Read `results/<dataset>/metrics.json` (mIoU / Dice, n=600). Worst-case figures per domain from `preds/`.
- README: question, datasets, sizes, method, results up front.
- Do not merge `--preview`, ViT-S, or Curia-2 numbers into the headline table.

### Phase 6 — Curia-2 ablation (stretch, CT/MRI only)

- After Phase 5. Adapter `src/backbones/curia2_backbone.py`; rest unchanged.
- Debiasing on and off, same KiPA + ACDC episodes.
- Separate README section; do not merge into the headline table.

## 6. Risks to handle in code and write-up

1. Record KiPA tumor pixel-area; do not silently crop small lesions
2. If cropping is ever added, report it as a separate condition
3. ACDC stays LV cavity; do not switch to myocardium
4. DINOv3 weights are gated — copy ViT-L to Drive before GPU eval
5. ACDC: native NIfTI only (MedOtter/CREATIS, not WSL4MIS)
6. Drive holds **processed PNGs, ViT-L, results**; Colab raw is deleted; laptop need not hold volumes
7. One slice per patient
8. Local one-pair / `--preview` numbers are not headline results
9. Push GitHub **before** Colab; clone with `--recurse-submodules` or `run_insid3.py` cannot import INSID3
10. Hugging Face login required on Colab for KiPA and ACDC downloads

## 7. Repository structure

```
insid3-medical-benchmark/
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version           # 3.10
├── src/
│   ├── data/
│   │   ├── download_kvasir.py
│   │   ├── download_kipa.py
│   │   ├── download_acdc.py
│   │   ├── prepare.py         # download → 2D PNG cache
│   │   └── preprocess.py      # slice rules
│   ├── datasets/
│   │   ├── polyp_loader.py
│   │   ├── kidney_tumor_loader.py
│   │   └── cardiac_loader.py
│   ├── backbones/
│   │   ├── dinov3_backbone.py
│   │   └── curia2_backbone.py # extension only
│   ├── run_insid3.py
│   └── evaluate.py
├── tests/
│   ├── test_preprocess.py
│   ├── test_prepare.py
│   ├── test_loaders.py
│   ├── test_evaluate.py
│   ├── test_run_insid3_dry_run.py
│   └── test_insid3_forward.py
├── third_party/               # INSID3 git submodule
├── notebooks/
│   └── colab_eval.ipynb
├── results/
│   ├── tables/
│   ├── figures/
│   └── extension_backbone_ablation/
└── docs/
    └── data.md
```

`data/` is gitignored. Colab has no unique model path; `notebooks/colab_eval.ipynb` only clones and calls the CLI against Drive.
