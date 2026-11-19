from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[2]
subset = pd.read_csv(root / "reports" / "subset_children.csv", dtype={"participant_id": str})

KEEP_SITES = ["KKI", "NYU", "OHSU", "Peking_1", "Peking_2"]
main = subset[subset["site"].isin(KEEP_SITES)].copy()
main.to_csv(root / "reports" / "subset_main.csv", index=False)

print("children kept:", len(main))
print(main["label"].value_counts())

# is age balanced between the two groups, inside each site?
print("\nmean age by site and label")
print(main.pivot_table(index="site", columns="label", values="age", aggfunc="mean").round(2))

# is sex balanced?
print("\nsex by label")
print(pd.crosstab(main["gender"], main["label"]))

# do the voxel-size flags hit one group more?
print("\nvoxel flags by label")
print(main.groupby("label")[["flag_coarse_voxel", "flag_fine_voxel"]].sum())