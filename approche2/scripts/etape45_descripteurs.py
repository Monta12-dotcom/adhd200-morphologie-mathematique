import csv
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import morpho  # noqa: E402

RACINE = Path(__file__).resolve().parent.parent
PRE = RACINE / "data" / "processed" / "pretraitement"
SORTIE = RACINE / "data" / "processed" / "descripteurs.csv"


def une_passe():
    res = pd.read_csv(PRE / "resultats.csv", dtype={"subject_id": str})
    deja = set()
    if SORTIE.exists():
        deja = set(pd.read_csv(SORTIE, dtype={"subject_id": str})["subject_id"])
    a_faire = res[~res["subject_id"].isin(deja)]
    print(f"{len(deja)} déjà faits, {len(a_faire)} à calculer", flush=True)

    colonnes = None
    if SORTIE.exists():
        colonnes = list(pd.read_csv(SORTIE, nrows=0).columns)

    for n, (_, l) in enumerate(a_faire.iterrows(), start=1):
        nom = f"{l['site']}_{l['subject_id']}"
        t0 = time.time()
        try:
            b, m, f, vc = morpho.charger_enfant(PRE / f"{nom}_brain.nii.gz",
                                                PRE / f"{nom}_mask.nii.gz",
                                                PRE / f"{nom}_frontal.nii.gz")
            d = morpho.descripteurs(b, m, f, vc)
        except Exception as e:
            print(f"[{n}/{len(a_faire)}] {nom} ÉCHEC {type(e).__name__}: {e}", flush=True)
            continue
        ligne = {"subject_id": l["subject_id"], "site": l["site"],
                 **{k: float(v) for k, v in d.items()}}
        if colonnes is None:
            colonnes = list(ligne)
        nouveau = not SORTIE.exists()
        with open(SORTIE, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=colonnes)
            if nouveau:
                w.writeheader()
            w.writerow(ligne)
        print(f"[{n}/{len(a_faire)}] {nom} ok ({time.time() - t0:.0f} s)", flush=True)
    return len(a_faire)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    # --suivre : continue à tourner et calcule les nouveaux enfants au fur et à
    # mesure que le prétraitement (étape 3) les produit
    suivre = "--suivre" in sys.argv
    while True:
        n = une_passe()
        if not suivre:
            break
        if n == 0:
            time.sleep(60)


if __name__ == "__main__":
    main()
