import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
DROP = {"vent_ok", "thr_csf_gm", "thr_gm_wm"}
QUALITY = ["thr_gm_wm", "thr_csf_gm", "tex_wm_int_std", "tex_gm_int_std"]   # contrast / noise proxies


def cohen_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / (sp + 1e-12))


def zsite(df, cols):
    g = df.groupby("site")[cols]
    return (df[cols] - g.transform("mean")) / (g.transform("std") + 1e-9)


def resid(Y, C):
    C = (C - C.mean(0)) / (C.std(0) + 1e-9)
    return Y - LinearRegression().fit(C, Y).predict(C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join("reports", "pipeline_all", "features_all.csv"))
    ap.add_argument("--out_dir", default=os.path.join("reports", "analysis"))
    ap.add_argument("--min_site_n", type=int, default=15, help="min children per class to report a site's own d")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    df = pd.read_csv(a.features, dtype={"participant_id": str})
    cols = list(df.columns)
    feats = [c for c in cols[cols.index("gm_frac"): cols.index("site")] if c not in DROP]
    df["sex"] = df["gender"].astype(str).str.lower().map({"male": 1, "m": 1, "female": 0, "f": 0})
    df = df[df["label"].isin(["TD", "ADHD"])].dropna(subset=["age", "sex", "ICV"])
    df = df[~np.isinf(df[feats]).any(axis=1)].reset_index(drop=True)
    feats = [c for c in feats if df[c].isna().mean() < 0.2 and df[c].std() > 0]
    qual = [q for q in QUALITY if q in df.columns and df[q].std() > 0]
    allc = list(dict.fromkeys(feats + qual))          # tex_* proxies are also features: no duplicates
    df[allc] = df[allc].fillna(df[allc].median())
    y = (df["label"] == "ADHD").values
    print(f"{len(df)} children ({y.sum()} ADHD), {len(feats)} features, quality proxies: {qual}")

    Z = zsite(df, allc)
    cov = df[["ICV", "age", "sex"]].values.astype(float)
    R = pd.DataFrame(resid(Z[feats].values, cov), columns=feats)
    Rq = pd.DataFrame(resid(Z[qual].values, cov), columns=qual) if qual else None
    Radj = pd.DataFrame(resid(Z[feats].values, np.c_[cov, Rq.values]), columns=feats) if qual else None
    sites = [s for s in df["site"].unique()
             if min((y & (df["site"] == s)).sum(), (~y & (df["site"] == s)).sum()) >= a.min_site_n]

    rows = []
    for f in feats:
        x = R[f].values
        d, p = cohen_d(x[y], x[~y]), ttest_ind(x[y], x[~y], equal_var=False).pvalue
        ds = {}
        for s in sites:
            k = (df["site"] == s).values
            ds[s] = cohen_d(x[k & y], x[k & ~y])
        row = {"feature": f, "d": d, "p": p,
               "sites_same_sign": int(sum(np.sign(v) == np.sign(d) for v in ds.values())), "n_sites": len(sites)}
        if qual:
            xa = Radj[f].values
            row["d_adj_quality"] = cohen_d(xa[y], xa[~y])
            row["p_adj_quality"] = ttest_ind(xa[y], xa[~y], equal_var=False).pvalue
            for q in qual:
                row[f"rho_{q}"] = spearmanr(R[f], Rq[q])[0]
        row.update({f"d_{s}": v for s, v in ds.items()})
        rows.append(row)
    T = pd.DataFrame(rows)
    T["p_fdr"] = multipletests(T["p"], method="fdr_bh")[1]
    if qual:
        T["p_fdr_adj_quality"] = multipletests(T["p_adj_quality"], method="fdr_bh")[1]
    T = T.sort_values("p")
    T.to_csv(os.path.join(a.out_dir, "univariate.csv"), index=False)

    if qual:
        Q = pd.DataFrame([{"quality_proxy": q, "d_ADHD_minus_TD": cohen_d(Rq[q].values[y], Rq[q].values[~y]),
                           "p": ttest_ind(Rq[q].values[y], Rq[q].values[~y], equal_var=False).pvalue} for q in qual])
        Q.to_csv(os.path.join(a.out_dir, "univariate_quality.csv"), index=False)
        print("\nDo the quality proxies differ between ADHD and controls? (after site z-score, ICV/age/sex removed)")
        print(Q.round(3).to_string(index=False))

    show = ["feature", "d", "p", "p_fdr", "sites_same_sign", "n_sites"] + \
           (["d_adj_quality", "p_fdr_adj_quality"] + [f"rho_{q}" for q in qual[:2]] if qual else [])
    print("\nTop 15 features by p-value (d > 0: higher in ADHD):")
    print(T[show].head(15).round(3).to_string(index=False))
    n_sig = int((T["p_fdr"] < 0.05).sum())
    print(f"\n{n_sig}/{len(T)} features significant at FDR 5%"
          + (f"; after adjusting for quality proxies: {int((T['p_fdr_adj_quality'] < 0.05).sum())}" if qual else ""))
    print(f"largest |d| = {T['d'].abs().max():.2f}  (literature: 0.1-0.2)")
    print(f"saved: {os.path.join(a.out_dir, 'univariate.csv')}")


if __name__ == "__main__":
    main()