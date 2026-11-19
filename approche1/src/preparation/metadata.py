from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "ADHD200_BIDS"
MISSING = ["-999", "N/A", "n/a", ""]



def deduplicate(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per (site, participant_id). Safe only because duplicates were
    verified to agree on every non-missing value (one L/Left spelling aside)."""
    meta = meta.copy()
    meta["handedness"] = meta["handedness"].replace({"L": "Left", "R": "Right"})
    return meta.groupby(["site", "participant_id"], as_index=False).first()

def load_participants(root: Path = DATA_ROOT) -> pd.DataFrame:
    frames = []
    for tsv in sorted(root.glob("*/participants.tsv")):
        try:
            df = pd.read_csv(tsv, sep="\t", dtype=str, na_values=MISSING,
                             keep_default_na=False, encoding="utf-8")
        except UnicodeDecodeError:
            print(f"[warn] {tsv.parent.name}: not UTF-8, falling back to latin-1")
            df = pd.read_csv(tsv, sep="\t", dtype=str, na_values=MISSING,
                             keep_default_na=False, encoding="latin-1")
        df.insert(0, "site", tsv.parent.name)
        df["participant_id"] = df["participant_id"].str.strip().str.zfill(7)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    meta = load_participants()
    inv = pd.read_csv(PROJECT_ROOT / "reports" / "t1w_inventory.csv", dtype={"participant_id": str})

    print(f"{len(meta)} metadata rows, {meta['participant_id'].nunique()} unique IDs")
    print("\nRows per site:\n", meta.groupby("site").size())
    print("\nColumns per site (non-empty counts):")
    print(meta.groupby("site").count().T.to_string())
    print("\ndx values per site:")
    print(meta.groupby(["site", "dx"], dropna=False).size().to_string())

    merged = inv.merge(meta, on=["site", "participant_id"], how="outer", indicator=True)
    print("\nScan/metadata match:\n", merged["_merge"].value_counts())
    print("\nScans with no metadata row, by site:\n",
          merged[merged["_merge"] == "left_only"].groupby("site").size())
    print("\nMetadata rows with no scan, by site:\n",
          merged[merged["_merge"] == "right_only"].groupby("site").size())