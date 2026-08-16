# INSID3 Cross-Domain Medical Segmentation Stress Test

## 1. Research Question

Three questions, in this order. They are not the same experiment.

**Q1 — Does the published medical pattern hold, and does INSID3 still lead
among training-free 1-shot methods?** The paper reports ISIC 54.4% mIoU vs.
chest X-ray lung 78.8% mIoU, and beats Matcher / GF-SAM on those sets. This
project repeats the same untuned 1-shot protocol on three new domains that
cross **modality** (endoscopy / CT / MRI) and **target type** (appearance-defined
lesion vs. well-outlined anatomy vs. small focus). “Wins” is defined only
against Matcher and GF-SAM on the **same episodes**. An INSID3-only table
cannot answer it.

**Q2 — Is the frozen natural-image encoder the limit?** Swap DINOv3 for
FlexiCT-2D (DINOv3 SSL recipe on a CT corpus) on KiPA, debias on and off.
A gain here is evidence that medical pretraining helps inside the same
INSID3 pipeline. A null result says the bottleneck is elsewhere (prompt,
clustering, resolution).

**Q3 — What does labelled medical supervision still buy?** Compare those
training-free 1-shot numbers to **box-prompted MedSAM2** zero-shot on the
same 2D slices. This is a supervision-gap reference, not a baseline:
different prompt, trained on medical masks. A MedSAM2 lead is a finding,
not a failure of Q1.

Headline write-up: Q1 table (INSID3 + Matcher + GF-SAM) plus qualitative
failures. Q2 and Q3 are required follow-up tables, never merged into one
flat ranking.

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
| DINOv2 ViT-L + SAM ViT-H | Drive `pretrain/` (Phase 6) | Public; same two files for Matcher and GF-SAM |
| FlexiCT-2D teacher | Drive `pretrain/flexict_2d_teacher.pth` (Phase 7) | CC BY-NC-ND 4.0 |
| MedSAM2 | Drive `pretrain/MedSAM2_latest.pt` (Phase 8) | From their `download.sh` |
| DINOv3 ViT-S | Laptop `pretrain/` (optional) | Local one-pair pytest only |
| `results/` | Drive `insid3-medical-benchmark/results/` | Colab disk is ephemeral |

Laptop keeps the git checkout, tests, and optionally ViT-S. It does **not** need `data/raw/` or the three-domain PNG cache.

## 4. Local tests, Colab data + GPU

Slice rules, loaders, metrics, and the CLI are implemented and tested on the laptop with **synthetic** fixtures (Phase 0–2). Colab is the worker for **real** download/preprocess and for headline GPU eval. The notebook (`notebooks/eval_insid3.ipynb`) has no unique model code: it clones the repo and calls `prepare.py`, `src/run_insid3.py` (Phase 3 shim), and `python -m src.methods.{gfsam,matcher,insid3,medsam2}` against Drive.

| Laptop | Colab Pro |
|---|---|
| `uv sync --group dev` + pytest | Clone + `%pip` project deps (runtime already has torch) |
| Optional ViT-S one-pair (`--model-size small --preview 1 --device cpu`) | `prepare.py` once → Drive `processed/`, delete raw |
| No KiPA/ACDC volumes required | 600 episodes × 3 domains, ViT-L, GPU |
| Push to GitHub before Colab | No pytest in the notebook |

## 5. Implementation Phases

### Phase 0 — Env and tests (laptop, no full datasets) — done when `uv run pytest` is green

- `uv` + `pyproject.toml` (Python 3.10 locally). Colab uses the runtime Python and installs the `methods` extra from the notebook, not vendor `requirements.txt` (`package = false`, CPython pin is 3.10).
- Pytest on **synthetic fixtures** (pairing, KiPA label 4, ACDC ED + LV, 600 random 1-shot, mIoU/Dice, `--dry-run`).
- Must pass with no GPU and no dataset download.

### Phase 1 — Push the worker, then build the 2D cache on Colab

Laptop:

1. Commit and push (include `.gitmodules` and every `third_party/*` gitlink).
2. Copy ViT-L to Drive: `MyDrive/insid3-medical-benchmark/pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`.
3. You may delete local `data/` to free disk. Keep ViT-S only if you still want the local one-pair test.

Colab (notebook section 1, CPU is enough for this step):

1. Open `notebooks/eval_insid3.ipynb` from the GitHub copy (File → Open notebook from GitHub) or upload it. GPU optional here.
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

Notebook: `notebooks/eval_insid3.ipynb`. Same flags as a local CLI; Drive paths only.

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

Section 2 of the notebook is a disposable GPU worker (one `DATASET` at a time; cell output is overwritten). Section 3 is the published report: it reads **all** `metrics.json` files, does not use `DATASET`, and must be re-run **once** after the three domains exist, then the notebook saved. GitHub/nbviewer keep that table, IoU histograms, and best/worst overlays.

### Phase 4 — KiPA resolution ablation

KiPA only. DINOv3 ViT-L/16 is patch-16: `--image-size 768` is a 48×48 token grid,
`1024` is 64×64. Kidney tumors are small foci on a full axial abdomen slice, so
a lesion can occupy a few patches at 768. That makes a low KiPA mIoU ambiguous
(method vs. token resolution). `slice_stats.json` records tumor pixel area so
failures can be checked against size. Polyps already fill much of a Kvasir frame;
ACDC ED LV is a large central cavity — neither is a small-object / few-token
problem, so a 1024 rerun there would not change the interpretation.

- Same 600 KiPA episodes at `--image-size 1024` only after Phase 3 used 768.
  Full-image, no crop. Write to a distinct output dir (e.g.
  `results/kidney_tumor_1024/`) so 768 is not overwritten.

### Phase 5 — Evaluate and write up

- Read `results/<dataset>/metrics.json` (mIoU / Dice, n=600). Worst-case figures per domain from `preds/`.
- Notebook section 3 is the qualitative write-up (table vs paper ISIC/CXR, histograms, area-vs-IoU, best/worst). Re-run it once, save the `.ipynb`.
- README: question, datasets, sizes, method, results up front.
- Do not merge `--preview`, ViT-S, or Phase 6–8 numbers into the INSID3-only Q1
  snapshot. After Phase 6, the Q1 table is INSID3 + Matcher + GF-SAM.

---

## Phases 6–8 — required (after Phase 5)

These are part of the project, not optional extras. Every run reuses the
**same episodes** as Phase 3 (`results/<dataset>/episodes.json`) and the same
metric code. Nothing here changes INSID3's algorithm — only the method
(Phase 6, 8) or the encoder (Phase 7).

| Phase | What | Answers |
|---|---|---|
| **6 (required)** | GF-SAM **and** Matcher on all three domains | Q1: does INSID3 still lead among training-free 1-shot methods? |
| **7 (required)** | FlexiCT-2D on KiPA, debias on/off | Q2: does a CT-pretrained encoder raise 1-shot mIoU? |
| **8 (required)** | MedSAM2 zero-shot, box-from-GT, all three domains | Q3: supervision gap vs training-free 1-shot |

Phase 7 ACDC (out-of-modality FlexiCT probe) stays secondary. Do not start
Phase 7 until Phase 6 has numbers — adapter debugging is easy to start and
hard to stop, and Q1 is incomplete without Matcher / GF-SAM.

### Comparison tiers

Results are reported in tiers, never in one flat ranking. The tier is
set by **what supervision went into the model before it ever saw these
datasets**, not by whether you fine-tuned anything.

| Tier | Method | Pretraining supervision | Prompt |
|---|---|---|---|
| **A** | INSID3 + DINOv3 (Phase 3 headline) | self-supervised, natural images | 1-shot mask |
| **A** | INSID3 + FlexiCT-2D (Phase 7) | self-supervised, CT corpus | 1-shot mask |
| **B** | Matcher, GF-SAM (Phase 6) | SAM decoder trained on SA-1B masks | 1-shot mask |
| **C** | MedSAM2 (Phase 8) | supervised on labelled medical masks | box / point |

Tier B is the comparison that makes Phase 3 readable: same one-shot mask
paradigm, no medical labels. Tier A (FlexiCT) is the same pipeline with the
encoder swapped. Tier C is a **reference point, not a baseline** — different
prompt and trained on medical masks — so it is never placed in the same row as
A or B.

### Shared implementer contract (Phases 6–8)

Do this once. Every Phase 6–8 script must follow it. Do **not** add
`--run-name` to any CLI (the existing `run_insid3.py` convention is
`--output-dir/<dataset>/` only).

**Episodes.** Never re-sample. Load Phase 3
`results/<dataset>/episodes.json` (seed 0, n=600). That JSON stores
`episode_index`, `reference_id`, `target_id`, and absolute paths. Resolve
paths in this order:

1. Use the stored path if the file still exists (same Drive mount).
2. Else rebuild from `--input-dir/<dataset>/{images,masks}/` via
   `load_paired_index` and the stored ids.
3. Fail loudly if an id is missing. Do not silently drop episodes.

Add `src/protocol/episodes_io.py` with `load_persisted_episodes(path,
input_dir) -> list[Episode]`. Reuse `src.evaluate.binary_iou` /
`binary_dice` / `mean_metrics`. Put `as_binary_mask` and
resize-to-GT in `src/protocol/masks.py` (re-export from
`src/evaluate.py` if tests already import it). The shared episode
loop lives in `src/protocol/loop.py` — every method CLI calls it
instead of copying `run_insid3.py`.

**Skip-if-complete.** If `metrics.json` exists and
`n == len(episodes)` and `seed`, `model`/`method`, and `image_size`
match the flags, skip that domain. Same idea as the Phase 3 notebook
eval cell.

**`metrics.json` is scores + protocol knobs only.** Keep anything
that *defines the number* (which episodes, which method, which
canvas, which prompt). Do **not** put Python / torch / numpy /
submodule SHAs here. Those do not change mIoU and they bloat every
table-loader. Phase 3 files already follow this; do not retrofit
them.

```text
dataset, protocol ("insid3_random"), seed, shots, n, mIoU, Dice,
items[{episode_index, reference_id, target_id, IoU, Dice}],
pred_dir,
method,          # "gfsam" | "matcher" | "insid3" | "medsam2"
image_size,      # canvas the method actually used
backbone,        # Phase 7: "dinov3" | "flexict2d"
debiased,        # Phase 7 only
prompt,          # "1shot_mask" | "box_from_gt"
tier             # "A" | "B" | "C"
```

**`run.json` (sibling, written by `loop.py` once per output dir).**
Machine / install provenance only. Notebook section 3 ignores it.

```text
python, torch, torchvision, numpy, cuda (or null),
git_head,                          # this repo
submodules: {INSID3, GF-SAM, Matcher, FlexiCT, MedSAM2}  # SHAs
```

One `run.json` per method×dataset directory is enough (same
granularity as `metrics.json`). Do not write a global
`results/_env.json` — a later Matcher-only rerun would lie about
the older GF-SAM numbers.

**Output roots on Drive** — method first, then dataset. Do **not**
write into `results/polyp/` etc.; those are the Phase 3
INSID3+DINOv3 snapshot. There is no `extensions/` folder.

```text
results/gfsam/<dataset>/
results/matcher/<dataset>/
results/flexict2d/debiased/<dataset>/
results/flexict2d/nodebiased/<dataset>/
results/medsam2/<dataset>/
```

Each of those dirs gets `metrics.json`, `run.json`, `episodes.json`
(copy of the Phase 3 list), and
`preds/{episode_index:04d}_{target_id}.png`.

**Image-size policy (locked).** Do not silently mix canvases.

| Method | Canvas | Why |
|---|---|---|
| INSID3 (already run) | 768 | Phase 3 headline |
| GF-SAM | **1024** | `main_eval.py` default `--img-size 1024`; SAM predictor canvas. DINOv2 is **always** resized to 518 inside `GFSAM.encoder_transform` (`patch_size=14`) |
| Matcher | **518** | `main_oss.py` default. Matcher uses the same size for DINOv2 and SAM. `feat_size = img_size // 14`, so the canvas **must** be divisible by 14. 768 is not (`768/14 ≈ 54.86`). Do not run Matcher at 768 |
| FlexiCT-2D (Phase 7) | 768 | same episodes / same INSID3 resize as Phase 3 |
| MedSAM2 (Phase 8) | 512 | MedSAM2 cfg `sam2.1_hiera_t512.yaml`; box is in **original** pixel coords |

Record `image_size` in every `metrics.json`. Resize the predicted mask
back to the **original GT PNG shape** with nearest-neighbour before
`binary_iou`. Never score on the model canvas.

**Host environment (one env, Colab and local GPU).** Official
READMEs pin mutually exclusive stacks (Matcher/GF-SAM torch 1.13 +
numpy 1.22; INSID3 Python 3.10 + torch 2.7; FlexiCT Python 3.11 +
torch 2.8; MedSAM2 Python 3.12 + torch 2.5). Do **not** recreate
those envs. Inference code we call is ordinary PyTorch. Publish
**one** extra that both Colab and a local GPU user can install.

Locked host contract:

| Item | Pin | Why |
|---|---|---|
| Python | **3.10** (already `.python-version`) | INSID3 `==3.10.*`. FlexiCT's 3.11 and MedSAM2's 3.12 are their conda recipes, not a syntax floor for the 2D paths we use |
| torch / torchvision | **≥2.5, <2.9** | MedSAM2 `setup.py` floor is 2.5.1. INSID3 and FlexiCT-2D run on 2.x. Matcher/GF-SAM INSTALL says **≥1.13.1** (the `==1.13.1` in their `requirements.txt` is a 2023 snapshot) |
| numpy | **≥2.0, <3** | INSID3 and MedSAM2 already want 2.x. Matcher uses removed `np.int` — monkeypatch in `src/methods/matcher/build.py` (`np.int = np.int64` if missing). Do not pin numpy 1.22 |
| Vendor install | **`sys.path` only** | Never `pip install -r third_party/*/requirements.txt`. Never `pip install segment-anything`. Never `pip install -e third_party/MedSAM2` unless `SAM2_BUILD_CUDA=0` (see Phase 8) |

`pyproject.toml` extras (implement when wiring Phase 6):

```toml
[project.optional-dependencies]
torch = [
    "torch>=2.5,<2.9",
    "torchvision>=0.20",
    "pycocotools>=2.0",
]
methods = [
    "opencv-python-headless>=4.8",
    "pot>=0.9",
    "omegaconf>=2.3",
    "iopath>=0.1.10",
    "timm>=0.9",
    "scipy>=1.11",
    "hydra-core>=1.3",
]
```

Local GPU user:

```bash
git clone --recurse-submodules <this-repo-url>
uv sync --extra torch --extra methods --group dev
# If `torch.cuda.is_available()` is False, install a CUDA wheel that
# matches the machine, then re-run the sync. Do not edit our pins to
# force cu118 vs cu124 — that is the user's driver, not the project.
```

Colab (runtime already has torch ≥2.5 on current Pro images):

```bash
git clone --recurse-submodules <this-repo-url>   # or submodule update --init
# do NOT pip-install torch
pip install opencv-python-headless pot omegaconf iopath timm scipy hydra-core
```

Write that provenance to `run.json` next to `metrics.json`, not
into the scores file. A second runtime is allowed only if a smoke
import actually fails.

**Package-name collision.** GF-SAM and Matcher both ship top-level
`matcher/`, `dinov2/`, and `segment_anything/`. Never put both roots
on `sys.path` in one process. Separate modules
(`python -m src.methods.gfsam` vs `python -m src.methods.matcher`).
Use **their** vendored `segment_anything`, not
`pip install segment-anything`. Matcher's
`SamAutomaticMaskGenerator` takes extra kwargs
(`sel_stability_score_thresh`, `dense_pred`, `sel_output_layer`, …)
that stock SAM does not accept.

---

### Phase 6 — Training-free baselines: Matcher and GF-SAM (tier B, required)

Do this **before** FlexiCT. A number like "0.41 mIoU on kidney tumor"
means little until the same 600 episodes are run through the methods
INSID3 itself compared against.

Sources: [ANDYZAQ/GF-SAM](https://github.com/ANDYZAQ/GF-SAM) (NeurIPS
2024, *Bridge the Points*), [aim-uofa/Matcher](https://github.com/aim-uofa/Matcher)
(ICLR 2024). Both are training-free 1-shot: DINOv2 ViT-L/14 for
correspondence + SAM ViT-H as the mask decoder.

Published 1-shot medical numbers from the INSID3 paper (Table 1):

| | Matcher (DINOv2+SAM) | GF-SAM (DINOv2+SAM) | INSID3 (DINOv3) |
|---|---|---|---|
| ISIC (diffuse lesion) | 38.6 | **48.7** | **54.4** |
| Chest X-ray (easy anatomy) | **70.8** | 51.0 | **78.8** |

GF-SAM is the stronger *average* training-free baseline; Matcher wins
on the paper's easy anatomical medical set. Those two published cells
are **anchors**, not descriptions of our domains. Do not call cardiac
"CXR-like" or polyp "ISIC-like" in the write-up — different organs and
modalities. **Do not run only GF-SAM and then claim INSID3 wins.**
Both methods, all three domains.

Runtime on an RTX 4090 (INSID3 paper): Matcher ~9 s/image, GF-SAM
~1 s, INSID3 ~0.3 s. Matcher × 3 × 600 is on the order of **4–5
hours**. Run GF-SAM first.

#### 6.1 Vendor the repos (git submodules, like INSID3)

Pin source commits in `.gitmodules`. This is independent of the
host env: we import from `third_party/<name>` via `sys.path`, we do
not `pip install` those trees.

From the laptop (once; commit the gitlinks):

```bash
git submodule add https://github.com/ANDYZAQ/GF-SAM.git third_party/GF-SAM
git submodule add https://github.com/aim-uofa/Matcher.git third_party/Matcher
git submodule update --init --recursive
```

Colab / a new local clone:

```bash
git clone --recurse-submodules <this-repo-url>
# already cloned:
git submodule update --init --recursive
```

Do **not** run `main_eval.py` / `main_oss.py`. Those scripts build
COCO/FSS/ISIC dataloaders (`FSSDataset`) and write their own loggers.
Our episodes are already PNG pairs. Wrap `set_reference` / `set_target`
/ `predict` / `clear` only.

GF-SAM's `--benchmark isic` (`scripts/isic.sh`) is an optional
**wiring check** on their ISIC loader. It is not our headline and
must not overwrite our `metrics.json`.

#### 6.2 Install (smoke, before booking a long GPU session)

Use the **host extra** in the shared contract (`--extra methods`).
Do not `pip install -r third_party/GF-SAM/requirements.txt` or
Matcher's — that would downgrade torch to 1.13 and numpy to 1.22
and break INSID3 / MedSAM2 / FlexiCT.

Both `INSTALL.md` files ask for Detectron2. The OSS path we use
(`matcher/GFSAM.py`, `matcher/Matcher.py`) does **not** import
Detectron2. Semantic-SAM does; we are not using Semantic-SAM
(`--use_semantic_sam` / `swint_only_sam_many2many.pth` is the
part-segmentation path — leave it off).

`src/methods/gfsam/build.py` / `matcher/build.py` insert **one**
vendor root on `sys.path` and import. Matcher `build.py` also
applies the NumPy 2 shim (`np.int` → `np.int64`).

Smoke, in separate processes (package names collide):

```bash
python -c "from src.methods.gfsam.build import build_gfsam; print('gfsam ok')"
python -c "from src.methods.matcher.build import build_matcher; print('matcher ok')"
```

If Detectron2 is missing and an import fails, you enabled the wrong
path — do not install Detectron2 "just in case."

#### 6.3 Shared weights on Drive

Both methods use the **same** two files. Download once to
`MyDrive/insid3-medical-benchmark/pretrain/`:

| File | URL |
|---|---|
| `dinov2_vitl14_pretrain.pth` | https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth |
| `sam_vit_h_4b8939.pth` | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |

SAM ViT-H is ~2.4 GB. Confirm sizes before the 600-episode loop.
CLI flags: `--dinov2-weights` / `--sam-weights` defaulting to
`pretrain/` (the Drive symlink already used for DINOv3).

`build_model` / `build_matcher_oss` construct DINOv2 as
`vits.vit_large` with `img_size=518`, `patch_size=14`, then
`dinov2_utils.load_pretrained_weights(..., "teacher")`, and SAM as
`sam_model_registry["vit_h"]`.

#### 6.4 Tensor contract (both repos, identical)

Read this against `GFSAM.set_reference` / `Matcher.set_reference` and
`main_eval.py` / `main_oss.py`. Their FSS batch is:

- `support_imgs`: `(B, nshot, 3, H, W)` in **[0, 1]** (not ImageNet-normalised)
- `support_masks`: `(B, nshot, H, W)` float/bool
- `query_img`: `(B, 3, H, W)` in **[0, 1]**

Inside `set_reference`:

```text
imgs = imgs.flatten(0, 1)                 # (B*nshot, 3, H, W)
masks = reference_masks_verification(masks)
masks = masks.permute(1, 0, 2, 3)         # (nshot, B, H, W)  — B must be 1
```

If the reference mask is all-zero they punch a **14×14 centre square**
to 1. Our processed masks should not be empty; if one is, log a
warning (that patch is their behaviour, not ours).

`set_target` asserts `H == W == input_size`, then

```text
img_np = img.mul(255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
```

for SAM. So the query **must** be float `[0, 1]`, not `uint8` and not
ImageNet-normalised. Each class applies ImageNet mean/std **inside**
`extract_img_feats` via `encoder_transform`.

Implement `src/protocol/fss_tensors.py` (Matcher / GF-SAM only):

```text
load_rgb_01(path, size) -> (3, size, size) float32 in [0, 1]
  PIL RGB, bilinear resize
load_mask_float(path, size) -> (size, size) float32 in {0, 1}
  nearest resize
pack_support(img, mask) ->
  support_imgs  (1, 1, 3, H, W)
  support_masks (1, 1, H, W)
pack_query(img) -> (1, 3, H, W)
```

Move tensors to the model device. After `predict`, squeeze to 2D,
nearest-resize to the **original** target-mask shape, then
`as_binary_mask` + `binary_iou`. Call `clear()` in a `finally` so a
failed episode cannot leak reference state.

#### 6.5 Implement `src/methods/gfsam/`

`src/methods/gfsam/build.py` puts `third_party/GF-SAM` on `sys.path`
and returns a `GFSAM` instance. `src/methods/gfsam/run.py` is the
CLI (`python -m src.methods.gfsam`). It only parses flags and calls
`src.protocol.loop.run_episodes`.

Flags: `--dataset`, `--input-dir`, `--output-dir` default
`results/gfsam`, `--episodes-json` (default
`<Drive>/results/<dataset>/episodes.json`), `--seed`, `--preview`,
`--device`, `--image-size` default **1024**, weight paths.

Do **not** call `python third_party/GF-SAM/main_eval.py`.

Build sequence:

1. `sys.path.insert(0, str(THIRD_PARTY_ROOT / "GF-SAM"))`.
2. Build a tiny `argparse.Namespace` with
   `dinov2_size="vit_large"`, `sam_size="vit_h"`, the two weight
   paths, `device=...`.
3. `from matcher.GFSAM import build_model, GFSAM`.
4. `model = build_model(args)`. **Note:** `build_model` does not pass
   `input_size`; `GFSAM` defaults to 1024. If you ever change
   `--image-size`, construct `GFSAM(encoder=..., generator=...,
   input_size=args.image_size, device=...)` yourself.
5. Per episode: pack tensors at 1024 → `set_reference` → `set_target`
   → `pred_masks, _ = model.predict()` → `model.clear()`.
6. `pred_masks` is `(1, 1024, 1024)` (or zeros of that shape when SAM
   returns no masks). Threshold `> 0`, resize to GT, score.

Empty-pred path in `GFSAM.predict` already returns
`torch.zeros((1, 1, 1024, 1024))`. Treat that as a valid all-background
mask (IoU 0 unless GT is also empty).

Stdout: do **not** dump the 600-item JSON into the notebook. Write
`metrics.json` and print only `dataset, n, mIoU, Dice` (same lesson as
Phase 3).

#### 6.6 Implement `src/methods/matcher/`

Same split: `build.py` (vendor `sys.path` + `build_matcher_oss`) and
`run.py` (`python -m src.methods.matcher`). `--output-dir` default
`results/matcher`, `--image-size` default **518**.

`sys.path` = `third_party/Matcher` only.

`build_matcher_oss(args)` needs the OSS flags from
[GETTING_STARTED.md](https://github.com/aim-uofa/Matcher/blob/main/GETTING_STARTED.md)
(these are **not** `main_oss.py` defaults — the paper/OSS recipe sets
them explicitly):

```text
--max_sample_iterations 64
--box_nms_thresh 0.65
--sample-range "(1,6)"
--topk_scores_threshold 0.0
--use_dense_mask 1
--use_points_or_centers
--purity_filter 0.02
--iou_filter 0.85
--multimask_output 1
--sel_stability_score_thresh 0.90
--use_score_filter
--alpha 1.0 --beta 0. --exp 0.
--num_merging_mask 9
```

Also set the remaining `main_oss.py` fields that `build_matcher_oss`
reads: `points_per_side=64`, `pred_iou_thresh=0.88`,
`stability_score_thresh=0.95`, `output_layer=3`,
`dense_multimask_output=0`, `num_centers=8`, `use_box=False`,
`emd_filter=0.0`, `coverage_filter=0.0`, `deep_score_filter=0.33`,
`deep_score_norm_filter=0.1`. Parse `--sample-range` with `eval` the
way they do, or pass a `(1, 6)` tuple directly.

`predict()` returns a float tensor mask, typically `(1, 518, 518)`.
Resize to GT.

**Failure isolation.** Matcher can throw on (a) empty Hungarian
assignment when no foreground DINOv2 tokens survive pooling, (b)
`ot.emd2` on an empty cost matrix, (c) `points` being a length-0
array in `patch_level_matching`. Catch per episode, write an
all-zero pred, record `IoU=0` / `Dice=0`, and continue. Log the
exception string on that `items[]` entry (`"error": "..."`). Do not
abort a 600-run for one pair.

`build_matcher` default `input_size=518` is also not taken from
`args.img_size`. If you change the flag, pass `input_size=` into
`Matcher(...)`.

#### 6.7 Smoke tests (mandatory before 600)

On Colab GPU, one domain (polyp is finest for a visual check):

```text
python -m src.methods.gfsam --dataset polyp --preview 1 \
  --episodes-json <Drive>/results/polyp/episodes.json \
  --input-dir <Drive>/processed \
  --output-dir <Drive>/results/gfsam \
  --dinov2-weights <Drive>/pretrain/dinov2_vitl14_pretrain.pth \
  --sam-weights <Drive>/pretrain/sam_vit_h_4b8939.pth
```

Then Matcher with `--preview 1`. Confirm:

- pred PNG is binary and **original** Kvasir size (not 1024/518)
- `metrics.json` has `n=1` and a finite IoU
- `clear()` was called (second episode must not see the first ref)

Then `--preview 8` on each method. Only then start 600.

#### 6.8 Full runs and notebook

Order: GF-SAM `polyp` → `kidney_tumor` → `cardiac`, then Matcher in
the same order. One method×domain per session if the VM dies; skip
logic covers restarts.

Add notebook cells under a new **section 2b (tier B)** that loops
`DOMAINS` and calls the two CLIs. Redirect stdout (Phase 3 lesson).
Section 3 (report) later grows a **separate** Q1 table:
INSID3 | Matcher | GF-SAM, one row per domain. Do not put those
numbers into the existing INSID3-only snapshot table.

#### 6.9 Phase 6 failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `SamAutomaticMaskGenerator() got unexpected keyword` | stock SAM on `sys.path` | only Matcher's `segment_anything/` |
| `module 'matcher' has no attribute 'GFSAM'` | Matcher root is on `sys.path` | one repo per process |
| `assert img_h == self.input_size` | query not resized to 1024 / 518 | `fss_tensors` size must match the class |
| pred shape 1024 vs GT 500+ | forgot resize-to-GT | nearest interpolate before `binary_iou` |
| Detectron2 missing | Semantic-SAM import leaked | do not enable `--use_semantic_sam` |
| Matcher hangs / huge RAM | EMD on many masks; expected to be slow | do not lower OSS flags to "go faster" for the headline table |
| all-zero preds | empty ref after pool, or SAM returned nothing | check ref mask load; 14×14 fallback should fire |
| `AttributeError: np.int` | NumPy 2 removed the alias | Matcher `build.py` shim; do not downgrade numpy |

---

### Phase 7 — FlexiCT-2D backbone swap (tier A, CT-primary, required)

Do **not** start until Phase 6 has Matcher + GF-SAM numbers on all
three domains. Adapter debugging is easy to start and hard to stop,
and Q1 is incomplete without tier B.

Source: [ricklisz/FlexiCT](https://github.com/ricklisz/FlexiCT)
(`Flexi_CT_2D`). Paper: Li et al., arXiv:2605.21906. Weights:
**CC BY-NC-ND 4.0** (research OK; note in README). Cite
`li2026universal`.

FlexiCT-2D is a 144M ViT (`embed_dim=864`, `depth=16`, `num_heads=12`,
`in_chans=1`, 4 register tokens) trained with the **DINOv3 SSL
recipe** on 266,227 CT volumes. It is **not** verifiable as
"domain-continued DINOv3" from the public repo. Write-up language:
"DINOv3 SSL recipe on a CT corpus."

#### 7.1 Vendor and checkpoint

```bash
git submodule add https://github.com/ricklisz/FlexiCT.git third_party/FlexiCT
```

Download FlexiCT-2D (`ct_2d_teacher.pth`, 144M) from the [Google Drive
link in their README](https://drive.google.com/file/d/1nUj2RCsNQfOAncMYY5S-YgQthteoAdSM/view?usp=drive_link)
to Drive `pretrain/flexict_2d_teacher.pth` (local users: `pretrain/`).

Their README wants Python 3.11 and torch 2.8. That is the demo
conda recipe (notebooks, MONAI, nnU-Net). The 2D wrapper is a ViT
forward. Add `third_party/FlexiCT` to `sys.path` and
`from flexi_ct import Flexi_CT_2D`. Do **not** `pip install -r`
their `requirements.txt`.

Smoke:

```python
from flexi_ct import Flexi_CT_2D
m = Flexi_CT_2D(checkpoint_path=".../flexict_2d_teacher.pth", device="cuda")
x = torch.zeros(1, 1, 256, 256, device="cuda")
out = m(x)
assert out["cls_token"].shape == (1, 864)
assert out["patch_tokens"].shape[-1] == 864
```

`Flexi_CT_2D.__init__` loads `ckpt["teacher"]`, strips the
`backbone.` prefix, drops `ibot` / `dino_head` keys, and
`load_state_dict(..., strict=True)`. If that raises, stop — do not
`strict=False` and run 600 episodes.

#### 7.2 Encoder contract (what INSID3 actually calls)

`INSID3._extract_features` does:

```text
fmaps = self.encoder.get_intermediate_layers(x, n=1, reshape=True)[0]
# x: (B, 3, image_size, image_size)  ImageNet-normalised
# fmaps: (B, C, h, w)
```

`_build_positional_basis` sends a **3-channel ImageNet-normalised
zero image** through the same method, then SVD; `svd_components=500`
requires `C ≥ 500`. FlexiCT `C=864`, so 500 is valid.

**The public wrapper does not implement this.** `Flexi_CT_2D.forward`
returns `{cls_token, patch_tokens}` only. `get_intermediate_layers`
lives on `Flexi_CT_2D.backbone` (`Flexi_CT_Backbone` in
`flexi_ct/models.py`): it takes last-`n` blocks, optionally norms,
drops CLS + 4 storage tokens, and if `reshape=True` returns
`(B, C, H/P, W/P)`.

Write `src/methods/insid3/flexict2d.py`:

```text
class FlexiCT2DEncoder(nn.Module):
    def __init__(self, checkpoint_path, patch_size=16, device="cuda"):
        self.model = Flexi_CT_2D(checkpoint_path=..., device=device)
        # released weights are patch_size=8; resample at runtime
        if patch_size != 8:
            self.model.backbone.patch_embed_2D.set_patch_size(patch_size)
            self.model.backbone.patch_size = patch_size
        # get_intermediate_layers reshape uses self.patch_size (isotropic int)

    def get_intermediate_layers(self, x, n=1, reshape=True):
        x1 = imagenet3_to_flexict1(x)          # (B, 1, H, W)
        return self.model.backbone.get_intermediate_layers(
            x1, n=n, reshape=reshape
        )
```

**Patch size: start at 16.** Default 8 at `--image-size 768` is a
96×96 grid (~9.2k tokens). INSID3 agglomerative clustering will be
slow and may OOM. `set_patch_size(16)` resamples the conv kernel
without changing checkpoint shapes (their README). 768 is divisible
by 8 and 16. Record `flexict_patch_size` in `metrics.json`. Only try
patch 8 if 16 is a null result and VRAM allows.

#### 7.3 Input conversion (locked)

Do **not** feed 3-channel ImageNet tensors into `in_chans=1`. Do
**not** replicate gray to 3 channels.

INSID3 `build_transform` is always RGB + ImageNet. Do not edit the
submodule. Convert **inside the adapter** so positional-basis zeros
and real images share one path:

```text
imagenet3_to_flexict1(x):          # x (B, 3, H, W) ImageNet-normalised
    rgb01 = x * std + mean         # undo ImageNet, clamp [0, 1]
    gray01 = rgb01.mean(dim=1, keepdim=True)
    return gray01 * 2 - 1          # FlexiCT demo range [-1, 1]
```

Their `inference_demo.ipynb` uses `hu_normalize`: clamp HU to
`[-1000, 1000]`, map to `[-1, 1]`. Our KiPA cache is already
**1–99 percentile uint8 PNGs** (`slice_to_uint8`). True HU is gone
(raw NIfTI was deleted). The adapter therefore approximates:
uint8 → [0, 1] → [-1, 1]. Write
`input_norm: "png_uint8_to_minus1_1"` in `metrics.json`. Do not
re-download KiPA volumes for this phase unless the swap is a null
and you have reason to blame the window.

#### 7.4 Debias on/off (INSID3 always debiases today)

`INSID3.predict_mask` always runs `_debias_features`. There is no
flag. **Do not edit the submodule.** Local subclass in
`src/methods/insid3/flexict2d.py`:

```text
class INSID3DebiasSwitch(INSID3):
    def __init__(..., use_debiased=True):
        self.use_debiased = use_debiased
        super().__init__(...)   # still builds the SVD basis

    def _debias_features(self, fmaps_norm):
        if not self.use_debiased:
            return fmaps_norm
        return super()._debias_features(fmaps_norm)
```

`--no-debiased` is the experiment, not a skip of SVD construction.

#### 7.5 Wire the INSID3 CLI

Keep `src/run_insid3.py` as a **thin shim** (`python src/run_insid3.py`
still works for the saved Phase 3 notebook). New flags and the
FlexiCT branch live in `src/methods/insid3/run.py` /
`src/methods/insid3/build.py`. Prefer
`python -m src.methods.insid3` for Phase 7 cells.

Add:

```text
--backbone {dinov3,flexict2d}     default dinov3
--flexict-weights PATH            default pretrain/flexict_2d_teacher.pth
--flexict-patch-size {8,16}       default 16
--debiased / --no-debiased        default on
--episodes-json PATH              required for Phase 7 (Phase 3 JSON)
```

When `backbone=flexict2d`, `build.py` uses `FlexiCT2DEncoder` instead
of `load_dinov3_encoder`, wraps `INSID3DebiasSwitch`, and ignores
`--model-size`. Keep `image_size=768`, `svd_comps=500`, τ/α unchanged.

`--output-dir` appends `--dataset` (same helper as today):

```text
--output-dir <Drive>/results/flexict2d/debiased
# → results/flexict2d/debiased/kidney_tumor/metrics.json
--output-dir <Drive>/results/flexict2d/nodebiased
```

Do not overwrite Phase 3 `results/kidney_tumor/`. JSON is the source
of truth for episodes; do not re-sample.

#### 7.6 Runs

Required:

1. KiPA, FlexiCT-2D, debias **on**, 600 episodes, 768, patch 16.
2. KiPA, FlexiCT-2D, debias **off**, same episodes.

Compare each to Phase 3 KiPA INSID3+DINOv3 (0.309 mIoU). Same
`src.evaluate` code.

Secondary, clearly labelled: ACDC with the same two flags. FlexiCT
is CT-pretrained; this is an **out-of-modality probe**. Polyp is
**out of scope**.

Smoke: `--preview 1` on KiPA, confirm
`get_intermediate_layers` returns `(1, 864, 48, 48)` at 768/16, then
`--preview 8`, then 600.

#### 7.7 Phase 7 failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: State-dict mismatch` | wrong ckpt or edited `_BACKBONE_KWARGS` | load released 2D teacher; do not change base `patch_size` before load |
| `expected 1 channel, got 3` | adapter not converting | convert inside `get_intermediate_layers` |
| OOM / clustering forever | still on patch 8 | `set_patch_size(16)` |
| `svd` / matmul dim error | `C < 500` | should be 864; you forwarded the wrapper not the backbone |
| reshape `h // patch_size` crash | `backbone.patch_size` not updated after resample | set both `patch_embed_2D` and `backbone.patch_size` |
| FlexiCT on polyp | out of scope | do not run |

---

### Phase 8 — MedSAM2 zero-shot reference (tier C, required)

Source: [bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2),
paper [arXiv:2504.03600](https://arxiv.org/abs/2504.03600). Checkpoint
via their `download.sh` → `checkpoints/MedSAM2_latest.pt`, cfg
`configs/sam2.1_hiera_t512.yaml`. Copy the `.pt` to Drive
`pretrain/MedSAM2_latest.pt`.

This is a **supervision-gap reference**, not a baseline. The prompt
is a box taken from the **target GT** — easier than a 1-shot
reference mask, and the model was trained on labelled medical masks.
A MedSAM2 lead is a finding. **Never put these numbers in the Q1
table.**

#### 8.1 What not to run

Official `medsam2_infer_3D_CT.py` is **3D video**:
`build_sam2_video_predictor_npz`, box on a key slice,
`propagate_in_video` forward and reverse, `getLargestCC`. That is
RECIST-to-volume, not our protocol.

Our cache is one 2D PNG per case. Report as **"box-prompted MedSAM2
on our 2D slices"**, not as a MedSAM2-paper reproduction. No
fine-tune. MedSAM3 is out of scope.

#### 8.2 Vendor and install (no CUDA extension)

```bash
git submodule add https://github.com/bowang-lab/MedSAM2.git third_party/MedSAM2
```

`src/methods/medsam2/build.py` puts `third_party/MedSAM2` on
`sys.path` and imports `sam2`. Install only the host extras
(`hydra-core`, `iopath`). Do **not** `pip install -e ".[dev]"`
(pulls Jupyter, tensorboard, …). Do **not** `pip install -e .`
without `SAM2_BUILD_CUDA=0`.

**What `SAM2_BUILD_CUDA` is.** MedSAM2 (SAM2) can compile
`sam2/csrc/connected_components.cu` into a native module
`sam2._C`. That extension is a small GPU connected-component
helper used in some SAM2 post-process paths. It is **not** the
Hiera decoder and **not** required for
`add_new_points_or_box` + `logits > 0` on a 1-frame state.

Compiling it is what breaks published installs:

- `pip install -e .` (default `SAM2_BUILD_CUDA=1`) invokes `nvcc`
  against **whatever torch is currently imported**.
- Colab often has no CUDA toolkit, or a toolkit that does not
  match the runtime torch wheel → build fails, or worse, a
  `_C.so` that later `import`s as
  `undefined symbol` / illegal instruction.
- A local user with torch-cu124 wheels and a system CUDA 11
  toolkit hits the same ABI crash. Default
  `SAM2_BUILD_ALLOW_ERRORS=1` then “succeeds” with a missing
  `_C`, and a later code path that *does* import it dies
  mid-run.

**Our rule:** never compile it. `sys.path` import does not run
`setup.py` at all. If someone insists on `pip install -e .`:

```bash
SAM2_BUILD_CUDA=0 pip install -e third_party/MedSAM2
```

Their README's Python 3.12 + `torch==2.5.1` is the authors'
training image. `setup.py` already allows `python>=3.10` and
`torch>=2.5.1`. Our host pin covers that. Do not invent a CPU
fallback if CUDA is missing — require a GPU, same as Phase 3.

#### 8.3 2D box protocol (`src/methods/medsam2/`)

Reuse Phase 3 episodes **only for the target slice** (reference image
is unused). Prompt:

```text
box = mask2D_to_bbox(gt, max_shift=0)   # copy from their script
# xyxy: x_min, y_min, x_max, y_max in original GT pixels
```

Their `mask2D_to_bbox` default `max_shift=20` **randomly expands**
the box. That is not acceptable for a deterministic table. Pass
`max_shift=0` (tight box). If GT is empty, write an all-zero pred
and continue.

Inference (1-frame "video", no propagate):

1. Load target PNG as RGB. Keep `orig_h, orig_w`.
2. Resize to 512×512, `/255`, ImageNet mean/std — same as
   `resize_grayscale_to_rgb_and_resize` + the norm in
   `medsam2_infer_3D_CT.py`. Tensor shape `(1, 3, 512, 512)` on CUDA.
3. `predictor = build_sam2_video_predictor_npz(cfg, ckpt)`.
4. `state = predictor.init_state(img, orig_h, orig_w)`.
5. Under `torch.inference_mode()` + `autocast` bfloat16:
   `_, _, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=state, frame_idx=0, obj_id=1, box=box)`.
6. `pred = (out_mask_logits[0] > 0.0).cpu().numpy()` and squeeze to
   2D. Their 3D script uses this threshold.
7. If pred is not `orig_h × orig_w`, nearest-resize to GT.
8. `predictor.reset_state(state)` every episode.

Do **not** call `propagate_in_video` (one frame; nothing to track).
Do **not** run `getLargestCC` unless you also report it as a separate
ablation — default off.

`src/methods/medsam2/box.py` owns `mask2D_to_bbox(..., max_shift=0)`.
`build.py` loads the predictor. `run.py` is
`python -m src.methods.medsam2` and uses `src.protocol.loop` (the
predict fn ignores the reference image).

CLI: `--dataset`, `--input-dir`, `--output-dir` default
`results/medsam2`, `--episodes-json`, `--checkpoint`, `--cfg`,
`--preview`, `--device`. `prompt=box_from_gt`, `tier=C` in JSON.

#### 8.4 Runs and notebook

All three domains, same 600 targets. KiPA first if install/VRAM
hurts. Smoke `--preview 1` then `8`, then 600.

Notebook: **section 2c (tier C)**, separate from 2 / 2b. Report
table is its own block: MedSAM2 mIoU/Dice vs the training-free 1-shot
number (INSID3, and optionally the better of Matcher/GF-SAM) with a
caption that the prompt and supervision are different.

#### 8.5 Phase 8 failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| boxes jitter across reruns | `max_shift>0` | force 0 |
| pred 512 vs GT native | forgot `init_state(..., orig_h, orig_w)` or forgot resize | score only at GT size |
| 3D NIfTI / RECIST code paths | copied `medsam2_infer_3D_CT.py` wholesale | 2D PNG + 1-frame state only |
| Q1 table includes MedSAM2 | reporting error | tier C only |
| CUDA extension build fail / `sam2._C` crash | `pip install -e` compiled `_C` against the wrong toolkit | never compile; `sys.path` import or `SAM2_BUILD_CUDA=0`. Do not silently CPU-eval 600 |

---

### Reporting rules

- One table per tier, in a separate README / notebook section from the
  INSID3-only snapshot.
- Never merge tier B/C numbers into the Phase 3 INSID3-only table. After
  Phase 6, the Q1 table is INSID3 + Matcher + GF-SAM.
- Do not ship the write-up without Phase 6 (all three domains, both methods),
  Phase 7 KiPA (debias on/off), and Phase 8 (all three domains). The only
  item that may slip is the Phase 7 ACDC probe.
- Do not describe cardiac as CXR-like or polyp as ISIC-like. Cite the
  paper's ISIC / CXR numbers as published anchors only.
- Phase 8 caption must say the prompt is a **target-GT box** and the
  model is medically supervised.

## 6. Risks to handle in code and write-up

1. Record KiPA tumor pixel-area; do not silently crop small lesions
2. If cropping is ever added, report it as a separate condition
3. ACDC stays LV cavity; do not switch to myocardium
4. DINOv3 weights are gated — copy ViT-L to Drive before GPU eval
5. ACDC: native NIfTI only (MedOtter/CREATIS, not WSL4MIS)
6. Drive holds **processed PNGs, all checkpoints, results**; Colab raw is deleted; laptop need not hold volumes
7. One slice per patient
8. Local one-pair / `--preview` numbers are not headline results
9. Push GitHub **before** Colab; clone with `--recurse-submodules` or `run_insid3.py` cannot import INSID3
10. Hugging Face login required on Colab for KiPA and ACDC downloads
11. Matcher / GF-SAM need SAM ViT-H + DINOv2 ViT-L on Drive; verify import (and Matcher's forked SAM) before a 600-episode run. Matcher ~9 s/image
12. MedSAM2 on this cache is 2D box-prompted slices (`max_shift=0`), not native 3D RECIST-to-volume
13. GF-SAM and Matcher both export `matcher` / `dinov2` / `segment_anything` — one `sys.path` root per process
14. Matcher canvas must be divisible by 14 (use 518). Do not run it at 768
15. FlexiCT-2D is 1-channel `[-1, 1]`; INSID3's transform is 3-channel ImageNet. Convert in the adapter. KiPA PNGs are percentile-stretched, not HU
16. FlexiCT public wrapper has no `get_intermediate_layers`; call `model.backbone`. Start at `patch_size=16`
17. Do not edit the INSID3 submodule for `--no-debiased`; subclass `_debias_features` in `src/methods/insid3/flexict2d.py`
18. Do not add `src/run_gfsam.py` / `run_matcher.py` / `run_medsam2.py` or a `results/extensions/` tree — use `src/methods/` and `results/<method>/`
19. One host env (Python 3.10, torch ≥2.5,<2.9, numpy 2.x + Matcher `np.int` shim). Never `pip install -r` a vendor `requirements.txt`. Never compile MedSAM2's `sam2._C`
20. All vendors are git submodules; clone with `--recurse-submodules`

## 7. Repository structure

Phases 6–8 are first-class methods, not an `extensions/` sidecar.
The layout has three layers: **data**, **protocol** (shared eval
contract), **methods** (one package per model). Vendored upstream
code never sits under `src/`. Results on Drive are keyed
`results/<method>/…`, not `results/extensions/…`.

### Layout rules

1. **Protocol vs method.** Episode reload, skip-if-complete, pred
   PNGs, `metrics.json`, and resize-to-GT live in `src/protocol/`.
   A method package only builds a model and predicts one mask.
2. **One package per method.** INSID3, GF-SAM, Matcher, and MedSAM2
   each own `src/methods/<name>/`. FlexiCT is an INSID3 encoder, so
   it lives under `methods/insid3/`, not a fifth top-level method.
3. **Invoke with `-m`.** `python -m src.methods.gfsam` (each package
   has `__main__.py` → `run.main`). Do not add a pile of
   `src/run_*.py` files. Keep `src/run_insid3.py` as a shim so the
   saved Phase 3 notebook still works.
4. **Do not move Phase 3 Drive paths.** `results/<dataset>/` is the
   INSID3+DINOv3 snapshot. New runs write beside it
   (`results/gfsam/polyp/`, …).
5. **Vendors stay in `third_party/` as git submodules** (INSID3,
   GF-SAM, Matcher, FlexiCT, MedSAM2). Never imported as `src.*`.
   `build.py` adds one vendor root to `sys.path` per process.
6. **One notebook.** `notebooks/eval_insid3.ipynb` remains the
   report. New cells call the method CLIs; they do not grow a second
   model tree.

### Source tree

```
insid3-medical-benchmark/
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version                 # 3.10
├── .gitignore                      # data/, pretrain/, results/  (vendors are submodules, not ignored)
├── src/
│   ├── data/                       # download + 2D cache (unchanged)
│   │   ├── prepare.py
│   │   ├── preprocess.py
│   │   ├── download_kvasir.py
│   │   ├── download_kipa.py
│   │   ├── download_acdc.py
│   │   ├── constants.py
│   │   ├── io_utils.py
│   │   └── paths.py
│   ├── datasets/                   # pairing + 1-shot sampling (unchanged)
│   │   ├── episodes.py
│   │   ├── pairing.py
│   │   ├── polyp_loader.py
│   │   ├── kidney_tumor_loader.py
│   │   └── cardiac_loader.py
│   ├── protocol/                   # shared eval contract (all methods)
│   │   ├── episodes_io.py          # load Phase 3 episodes.json + remap paths
│   │   ├── loop.py                 # skip-if-complete, predict, write metrics/preds
│   │   ├── masks.py                # as_binary_mask, resize_pred_to_gt
│   │   └── fss_tensors.py          # PNG → Matcher/GF-SAM [0,1] packs
│   ├── methods/
│   │   ├── insid3/                 # Q1 headline + Q2 encoder swap
│   │   │   ├── run.py              # CLI; --backbone {dinov3,flexict2d}
│   │   │   ├── build.py            # DINOv3 or FlexiCT2DEncoder → INSID3
│   │   │   ├── dinov3.py           # move of backbones/dinov3_backbone.py
│   │   │   └── flexict2d.py        # FlexiCT2DEncoder + INSID3DebiasSwitch
│   │   ├── gfsam/                  # Q1 / tier B
│   │   │   ├── run.py
│   │   │   └── build.py            # sys.path → third_party/GF-SAM
│   │   ├── matcher/                # Q1 / tier B
│   │   │   ├── run.py
│   │   │   └── build.py            # sys.path → third_party/Matcher
│   │   └── medsam2/                # Q3 / tier C
│   │       ├── run.py
│   │       ├── build.py
│   │       └── box.py              # tight GT box, max_shift=0
│   ├── run_insid3.py               # shim → methods.insid3.run (Phase 3 notebook)
│   └── evaluate.py                 # binary_iou / Dice; tests import this
├── tests/
│   ├── test_preprocess.py
│   ├── test_prepare.py
│   ├── test_loaders.py
│   ├── test_evaluate.py
│   ├── test_run_insid3_dry_run.py
│   ├── test_insid3_forward.py
│   ├── test_protocol_episodes.py   # JSON reload + path remap
│   ├── test_protocol_masks.py
│   └── test_medsam2_box.py         # max_shift=0, empty-GT
├── third_party/
│   ├── README.md                   # all five are git submodules; sys.path only
│   ├── INSID3/
│   ├── GF-SAM/
│   ├── Matcher/
│   ├── FlexiCT/
│   └── MedSAM2/
├── notebooks/
│   └── eval_insid3.ipynb           # setup, data, workers, Q1–Q3 report
├── plans/
│   └── plan.md
└── docs/
    └── data.md
```

`src/backbones/` goes away when Phase 7 is wired: move
`dinov3_backbone.py` → `src/methods/insid3/dinov3.py` and leave a
one-line re-export in `src/backbones/dinov3_backbone.py` so existing
tests keep importing. Do not add FlexiCT under `backbones/`.

Each `src/methods/<name>/` has `__main__.py` calling `run.main`, so:

```text
python -m src.methods.gfsam --dataset polyp ...
python -m src.methods.matcher --dataset kidney_tumor ...
python -m src.methods.insid3 --backbone flexict2d --dataset kidney_tumor ...
python -m src.methods.medsam2 --dataset cardiac ...
```

`loop.py` is the only place that writes `metrics.json` / `preds/`.
Method `run.py` files parse args, build the model, and pass
`predict_episode(ep) -> 2D mask` into the loop.

### Drive layout (not in git)

```
MyDrive/insid3-medical-benchmark/
├── processed/{polyp,kidney_tumor,cardiac}/{images,masks}/
├── pretrain/
│   ├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
│   ├── dinov2_vitl14_pretrain.pth
│   ├── sam_vit_h_4b8939.pth
│   ├── flexict_2d_teacher.pth
│   └── MedSAM2_latest.pt
└── results/
    ├── polyp/                      # Phase 3 INSID3+DINOv3 — do not overwrite
    ├── kidney_tumor/
    ├── cardiac/
    ├── kidney_tumor_1024/          # Phase 4 ablation
    ├── gfsam/{polyp,kidney_tumor,cardiac}/
    ├── matcher/{polyp,kidney_tumor,cardiac}/
    ├── flexict2d/
    │   ├── debiased/kidney_tumor/
    │   └── nodebiased/kidney_tumor/
    └── medsam2/{polyp,kidney_tumor,cardiac}/
```

Each method×dataset dir: `metrics.json`, `run.json`, `episodes.json`,
`preds/{episode_index:04d}_{target_id}.png`.

`--output-dir` is the method root. The loop appends `--dataset`
(`results/gfsam` + `polyp` → `results/gfsam/polyp/`). FlexiCT uses
the condition as the root (`results/flexict2d/debiased` +
`kidney_tumor`).

### What is gitignored

`data/`, `pretrain/`, `results/`. Vendor trees are submodules (like
INSID3), not ignored. A clone without `--recurse-submodules` must
run `git submodule update --init --recursive` or every method
`build.py` fails the same way Phase 3 failed without INSID3.

Colab has no unique model path. The notebook clones with
`--recurse-submodules`, symlinks Drive `processed/` / `pretrain/` /
`results/`, installs the host `methods` extra (not vendor
`requirements.txt`), and calls the method CLIs.
