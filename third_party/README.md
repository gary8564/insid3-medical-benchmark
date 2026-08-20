Vendored method code is **git submodules**. We import via `sys.path` only — never `pip install -r third_party/*/requirements.txt`, never `pip install segment-anything`, and never `pip install -e third_party/MedSAM2` unless `SAM2_BUILD_CUDA=0`.

```bash
# New clone
git clone --recurse-submodules <this-repo-url>

# Already cloned without submodules
git submodule update --init --recursive
```

| Folder | Used by | Notes |
|---|---|---|
| `INSID3/` | `python src/run_insid3.py` / `python -m src.methods.insid3` | Phase 3 + Phase 7 |
| `GF-SAM/` | `python -m src.methods.gfsam` | Package names collide with Matcher — separate process |
| `Matcher/` | `python -m src.methods.matcher` | Same collision; NumPy 2 `np.int` shim is in our `build.py` |
| `FlexiCT/` | `--backbone flexict2d` | Encoder only; do not install their conda stack |
| `MedSAM2/` | `python -m src.methods.medsam2` | Do not compile `sam2._C` |

DINOv3 / DINOv2 / SAM / FlexiCT / MedSAM2 **weights** stay in `pretrain/` (gitignored). Checkpoints are not part of the submodules.
