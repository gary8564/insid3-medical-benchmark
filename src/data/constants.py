"""Voxel class IDs and run-protocol defaults."""

# KiPA22: 1=renal vein, 2=kidney, 3=renal artery, 4=kidney tumor
KIPA_TUMOR_LABEL = 4

# ACDC: 1=RV cavity, 2=myocardium, 3=LV cavity
ACDC_LV_LABEL = 3

# INSID3 DatasetLung / DatasetISIC uses 600 random 1-shot episodes with seed 0
# (https://github.com/visinf/INSID3/blob/main/datasets/lung.py).
DEFAULT_SEED = 0
DEFAULT_SHOTS = 1
INSID3_N_EPISODES = 600

# Optional first-pass: randomly pick N episodes from the full sampled list (`--preview`).
PREVIEW_N = 8
