import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from boruta import BorutaPy
from scipy.stats import binom, pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
COV = ["ICV", "age", "sex"]          # covariates regressed out inside each fold
N_COV = len(COV)


# =========================================================================== transformers
class Residualizer(BaseEstimator, TransformerMixin):
    """Last n_cov columns = covariates. Returns features minus their linear
    prediction from the covariates. Fitted on the TRAIN fold only."""
    def __init__(self, n_cov=3):
        self.n_cov = n_cov

    def fit(self, X, y=None):
        self.lr_ = LinearRegression().fit(X[:, -self.n_cov:], X[:, :-self.n_cov])
        return self

    def transform(self, X):
        return X[:, :-self.n_cov] - self.lr_.predict(X[:, -self.n_cov:])


class BorutaSelector(BaseEstimator, TransformerMixin):
    """All-relevant selection (Boruta). Falls back to the best-ranked features if
    nothing is confirmed, so the pipeline never crashes."""
    def __init__(self, regression=False, max_iter=50, alpha=0.05, random_state=0, min_features=5):
        self.regression = regression
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state
        self.min_features = min_features

    def fit(self, X, y):
        if self.regression:
            est = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=self.random_state)
        else:
            est = RandomForestClassifier(n_jobs=-1, max_depth=5, class_weight="balanced",
                                         random_state=self.random_state)
        b = BorutaPy(est, n_estimators="auto", perc=100, alpha=self.alpha,
                     max_iter=self.max_iter, random_state=self.random_state)
        b.fit(np.asarray(X), np.asarray(y))
        sup = b.support_.copy()
        if sup.sum() == 0:
            sup = b.support_weak_.copy()
        if sup.sum() == 0:
            sup = np.zeros(X.shape[1], bool)
            sup[np.argsort(b.ranking_)[: self.min_features]] = True
        self.support_ = sup
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]


def make_pipe(regression, seed, boruta_iter, n_trees):
    if regression:
        rf = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    else:
        rf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=3, n_jobs=-1,
                                    class_weight="balanced", random_state=seed)
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("res", Residualizer(N_COV)),
        ("select", BorutaSelector(regression, boruta_iter, random_state=seed)),
        ("rf", rf),
    ])


# =========================================================================== data
FIRST_FEAT, LAST_FEAT = "gm_frac", "site"          # features sit between these two columns; "site" is the first metadata column
NEW_PREFIXES = ("oct_", "asym_", "tex_")             # pseudo-regional + texture features
DROP_FEATS = {"vent_ok", "thr_csf_gm", "thr_gm_wm"}      # bookkeeping values, not anatomy
SUBTYPE_C, SUBTYPE_I = "ADHD-Combined", "ADHD-Inattentive"


def load_data(a):
    df = pd.read_csv(a.features, dtype={"participant_id": str})
    cols = list(df.columns)
    feats = [c for c in cols[cols.index(FIRST_FEAT): cols.index(LAST_FEAT)] if c not in DROP_FEATS]
    if a.feature_set == "base":                      # the 47 global features
        feats = [c for c in feats if not c.startswith(NEW_PREFIXES)]
    elif a.feature_set == "new":                     # only the octant / asymmetry / texture features
        feats = [c for c in feats if c.startswith(NEW_PREFIXES)]
    if a.no_topology:                                # drop the noise-sensitive hole / fragment counts
        feats = [c for c in feats if not c.endswith(("_euler", "_ncc"))]
    df["sex"] = df["gender"].astype(str).str.strip().str.lower().map({"male": 1, "m": 1, "female": 0, "f": 0})
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["ICV"] = pd.to_numeric(df["ICV"], errors="coerce")
    for c in ("adhd_index", "inattentive", "hyper_impulsive", "med_status"):
        df[c] = pd.to_numeric(df[c], errors="coerce").mask(lambda x: x <= -900)   # -999 = missing
    df["dx"] = df["dx"].astype(str).str.strip()
    n0 = len(df)
    df = df[df["label"].isin(["TD", "ADHD"])].dropna(subset=["age", "sex", "ICV"])
    df = df[~np.isinf(df[feats]).any(axis=1)]
    feats = [c for c in feats if df[c].isna().mean() < 0.2 and df[c].std() > 0]
    print(f"{len(df)}/{n0} children kept, {len(feats)} features")
    print("diagnosis:", df["dx"].value_counts().to_dict())
    print("medication status (ADHD only):", df.loc[df["label"] == "ADHD", "med_status"].value_counts(dropna=False).to_dict())
    if a.site_norm:                                  # per-site z-score, label-free
        g = df.groupby("site")[feats]
        df[feats] = (df[feats] - g.transform("mean")) / (g.transform("std") + 1e-9)
    return df.reset_index(drop=True), feats


def get_task(df, task, a):
    d = df
    if a.exclude_medicated:
        naive = [float(v) for v in a.naive_values.split(",")]
        d = d[(d["label"] == "TD") | d["med_status"].isin(naive)]
    if task == "adhd_vs_control":
        y = (d["label"] == "ADHD").astype(int).values
    elif task == "subtype":                          # 1 = combined, 0 = inattentive (ADHD children only)
        d = d[d["dx"].isin([SUBTYPE_C, SUBTYPE_I])]
        y = (d["dx"] == SUBTYPE_C).astype(int).values
    else:
        col = {"reg_hyper": "hyper_impulsive", "reg_inatt": "inattentive", "reg_index": "adhd_index"}[task]
        d = d[d[col].notna()]
        y = d.groupby("site")[col].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9)).values
    return d, y


# =========================================================================== CV engine
def make_splits(y, groups, mode, n_rep, n_splits, regression, seed):
    idx = np.arange(len(y))
    if mode == "loso":
        return [list(LeaveOneGroupOut().split(idx, y, groups))]
    out = []
    for r in range(n_rep):
        cv = KFold(n_splits, shuffle=True, random_state=seed + r) if regression \
            else StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        out.append(list(cv.split(idx, y)))
    return out


def cv_run(X, y, reps, regression, a, seed=0):
    n, n_feat = len(y), X.shape[1] - N_COV
    oof = np.full((len(reps), n), np.nan)
    sel, n_it = np.zeros(n_feat), 0
    for ri, splits in enumerate(reps):
        for tr, te in splits:
            pipe = make_pipe(regression, seed + ri, a.boruta_iter, a.n_trees)
            pipe.fit(X[tr], y[tr])
            oof[ri, te] = pipe.predict(X[te]) if regression else pipe.predict_proba(X[te])[:, 1]
            sel[pipe.named_steps["select"].support_] += 1
            n_it += 1
    return oof, sel, n_it


def cohens_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / (sp + 1e-12))


def evaluate(y, oof, age, site, regression):
    rows = []
    for p in oof:
        ok = ~np.isnan(p)
        if regression:
            rows.append({"r": pearsonr(p[ok], y[ok])[0], "r_with_age": pearsonr(p[ok], age[ok])[0]})
        else:
            rows.append({"auc": roc_auc_score(y[ok], p[ok]),
                         "bal_acc": balanced_accuracy_score(y[ok], p[ok] > 0.5),
                         "cohen_d": cohens_d(p[ok][y[ok] == 1], p[ok][y[ok] == 0]),
                         "r_pred_vs_age": pearsonr(p[ok], age[ok])[0]})
    m = pd.DataFrame(rows)
    out = {k: [round(float(m[k].mean()), 3), round(float(m[k].std(ddof=0)), 3)] for k in m}
    per_site = {}                                  # first repeat, per site
    p = oof[0]
    for s in np.unique(site):
        k = (site == s) & ~np.isnan(p)
        if k.sum() < 15:
            continue
        if regression:
            per_site[s] = round(float(pearsonr(p[k], y[k])[0]), 3)
        elif min((y[k] == 1).sum(), (y[k] == 0).sum()) >= 5:
            per_site[s] = round(float(roc_auc_score(y[k], p[k])), 3)
    out["per_site"] = per_site
    return out


def selection_table(names, sel, n_it, a, X, y, regression, mode):
    tab = pd.DataFrame({"feature": names, "sel_freq": sel / max(n_it, 1)})
    if a.n_perm > 0 and mode == "kfold":
        rng = np.random.default_rng(a.seed)
        pf = np.zeros(len(names))
        for _ in range(a.n_perm):
            yp = rng.permutation(y)
            reps = make_splits(yp, None, "kfold", 1, a.n_splits, regression, int(rng.integers(1e6)))
            _, s, it = cv_run(X, yp, reps, regression, a)
            pf += s / it
        p0 = np.clip(pf / a.n_perm, 0.5 / (a.n_perm * a.n_splits), 1)
        tab["p_binom"] = binom.sf(sel - 1, n_it, p0)
        tab["p_holm"] = multipletests(tab["p_binom"], method="holm")[1]
    return tab.sort_values("sel_freq", ascending=False)


def run_task(df, feats, task, a):
    d, y = get_task(df, task, a)
    regression = task.startswith("reg_")
    X = np.c_[d[feats].values, d[COV].values]
    site, age = d["site"].values, d["age"].values
    print(f"\n=== {task} | n={len(y)}" + ("" if regression else f" | positives={int(y.sum())}"))
    results = []
    for mode in ("loso", "kfold"):
        reps = make_splits(y, site, mode, a.n_repeats, a.n_splits, regression, a.seed)
        oof, sel, n_it = cv_run(X, y, reps, regression, a, a.seed)
        met = evaluate(y, oof, age, site, regression)
        tab = selection_table(feats, sel, n_it, a, X, y, regression, mode)
        tag = f"{task}_{mode}" + ("_naive" if a.exclude_medicated else "") + a.tag
        tab.to_csv(os.path.join(a.out_dir, f"{tag}_features.csv"), index=False)
        res = {"task": task, "mode": mode, "n": int(len(y)), "medicated_excluded": a.exclude_medicated, **met}
        json.dump(res, open(os.path.join(a.out_dir, f"{tag}.json"), "w"), indent=2)
        print(mode.upper(), {k: v for k, v in met.items() if k != "per_site"})
        print("   per-site:", met["per_site"])
        print("   top features:", ", ".join(tab["feature"].head(5)))
        results.append(res)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join("reports", "pipeline_all", "features_all.csv"))
    ap.add_argument("--out_dir", default=os.path.join("reports", "analysis"))
    ap.add_argument("--naive_values", default="0", help="med_status values meaning medication-naive (check the printed counts)")
    ap.add_argument("--feature_set", choices=["all", "base", "new"], default="all",
                    help="all = global + octant + texture; base = the 47 global features; new = octant/asymmetry/texture only")
    ap.add_argument("--no_topology", action="store_true", help="drop *_euler and *_ncc (noise-sensitive counts)")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. _notopo")
    ap.add_argument("--tasks", default="adhd_vs_control,subtype,reg_inatt,reg_hyper")
    ap.add_argument("--exclude_medicated", action="store_true")
    ap.add_argument("--no_site_norm", dest="site_norm", action="store_false")
    ap.add_argument("--n_splits", type=int, default=10)
    ap.add_argument("--n_repeats", type=int, default=3)
    ap.add_argument("--boruta_iter", type=int, default=50)
    ap.add_argument("--n_trees", type=int, default=300)
    ap.add_argument("--n_perm", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    df, feats = load_data(a)
    allres = []
    for t in a.tasks.split(","):
        allres += run_task(df, feats, t.strip(), a)
    rows = []
    for r in allres:
        rows.append({"task": r["task"], "mode": r["mode"], "n": r["n"],
                     **{k: v[0] for k, v in r.items() if isinstance(v, list)}})
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(a.out_dir, "summary" + ("_naive" if a.exclude_medicated else "") + a.tag + ".csv"), index=False)
    print("\n", summ.to_string(index=False))


if __name__ == "__main__":
    main()