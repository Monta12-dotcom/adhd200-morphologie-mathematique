from pathlib import Path
import pandas as pd
from metadata import load_participants, deduplicate

ROOT = Path(__file__).resolve().parents[2]


def make_label(dx):
    if dx == "Typically Developing Children":
        return "TD"
    if isinstance(dx, str) and dx.startswith("ADHD"):
        return "ADHD"
    return "unknown"


def make_hand_class(value):
    if pd.isna(value):
        return "unknown"
    if value in ("Right", "Left", "Ambidextrous"):
        return value
    score = float(value)
    if score > 0:
        return "Right"
    if score < 0:
        return "Left"
    return "Ambidextrous"


meta = deduplicate(load_participants())
scans = pd.read_csv(ROOT / "reports" / "t1w_inventory.csv", dtype={"participant_id": str})
cohort = scans.merge(meta, on=["site", "participant_id"], validate="one_to_one")

cohort["age"] = pd.to_numeric(cohort["age"])
cohort["label"] = cohort["dx"].apply(make_label)
cohort["hand_class"] = cohort["handedness"].apply(make_hand_class)

cohort["voxel_max"] = cohort[["vox_x", "vox_y", "vox_z"]].max(axis=1)
cohort["voxel_min"] = cohort[["vox_x", "vox_y", "vox_z"]].min(axis=1)

cohort["flag_no_label"] = cohort["label"] == "unknown"
cohort["flag_coarse_voxel"] = cohort["voxel_max"] > 1.5
cohort["flag_fine_voxel"] = cohort["voxel_min"] < 0.9
cohort["flag_missing_sex"] = cohort["gender"].isna()
cohort["flag_age_over_18"] = cohort["age"] > 18

cohort.to_csv(ROOT / "reports" / "cohort.csv", index=False)

print("Rows:", len(cohort))
print()
print(cohort["label"].value_counts().to_string())
print()
print(cohort.groupby(["site", "label"]).size().unstack(fill_value=0).to_string())
print()
flags = [c for c in cohort.columns if c.startswith("flag_")]
print(cohort[flags].sum().to_string())
print()
print("Handedness classes:", cohort["hand_class"].value_counts().to_dict())
print()
print("Voxel sizes by site (min x max):")
print(cohort.groupby(["site", "voxel_min", "voxel_max"]).size().to_string())