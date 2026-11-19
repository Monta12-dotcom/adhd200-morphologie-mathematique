from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[2]
cohort = pd.read_csv(root / "reports" / "cohort.csv", dtype={"participant_id": str})
print("columns:", list(cohort.columns))

# keep only labelled participants with a known age
labelled = cohort[cohort["label"].isin(["ADHD", "TD"])].copy()
labelled["age"] = pd.to_numeric(labelled["age"], errors="coerce")
labelled = labelled.dropna(subset=["age"])

# how many children for each possible age cutoff?
for cutoff in [10, 12, 13, 14, 18]:
    kids = labelled[labelled["age"] <= cutoff]
    print("\n=== age <= " + str(cutoff) + ": " + str(len(kids)) + " children ===")
    print(pd.crosstab(kids["site"], kids["label"]))

# save the chosen subset (change CUTOFF after you look at the tables)
CUTOFF = 12
subset = labelled[labelled["age"] <= CUTOFF]
subset.to_csv(root / "reports" / "subset_children.csv", index=False)
print("\nsaved", len(subset), "children with age <=", CUTOFF)