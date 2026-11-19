from pathlib import Path
import re
import nibabel as nib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "ADHD200_BIDS"
# Only raw scans: exclude derived files (toCano_, siamV01_ prefixes).
RAW_PATTERN = re.compile(r"^sub-(\d+)_(ses-[^_]+)_(?:(acq-[^_]+)_)?(run-[^_]+)_T1w\.nii\.gz$")


def scan_t1w(root: Path = DATA_ROOT) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*_T1w.nii.gz")):
        m = RAW_PATTERN.match(path.name)
        if m is None:               # derived or unexpected name
            continue
        site = path.relative_to(root).parts[0]
        row = {
            "site": site,
            "participant_id": m.group(1),
            "session": m.group(2),
            "acq": m.group(3),
            "run": m.group(4),
            "size_mb": round(path.stat().st_size / 1e6, 2),
            "path": str(path),
        }
        try:
            img = nib.load(path)    # header only, no voxel data loaded
            zooms = img.header.get_zooms()[:3]
            row.update(
                shape=str(img.shape),
                ndim=len(img.shape),
                vox_x=round(float(zooms[0]), 3),
                vox_y=round(float(zooms[1]), 3),
                vox_z=round(float(zooms[2]), 3),
                orientation="".join(nib.aff2axcodes(img.affine)),
                error="",
            )
        except Exception as e:      # record failures, do not hide them
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = scan_t1w()
    out = PROJECT_ROOT / "reports" / "t1w_inventory.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} raw T1w files, {df['participant_id'].nunique()} unique participants")
    print(f"Header read errors: {(df['error'] != '').sum()}")
    print("\nFiles per site:\n", df.groupby("site").size())
    print("\nOrientations:\n", df["orientation"].value_counts(dropna=False))
    print("\nVoxel sizes (x,y,z):\n", df.groupby(["vox_x", "vox_y", "vox_z"]).size())
    print("\nSessions / acq:\n", df.groupby(["session", "acq"], dropna=False).size())