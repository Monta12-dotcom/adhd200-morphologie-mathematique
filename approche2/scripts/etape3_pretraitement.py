import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

RACINE = Path(__file__).resolve().parent.parent
MODELES = RACINE / "models"
TABLE = RACINE / "data" / "processed" / "table_propre.csv"
SORTIE = RACINE / "data" / "processed" / "pretraitement"

COLONNES = ["site", "subject_id", "adhd", "vol_cerveau_ml", "vol_frontal_ml",
            "dice_recalage", "duree_synthstrip_s", "duree_recalage_s"]


def synthstrip(entree, sortie_cerveau, sortie_masque):
    cmd = [sys.executable, str(MODELES / "lance_synthstrip.py"),
           "-i", str(entree), "-o", str(sortie_cerveau),
           "-m", str(sortie_masque), "--model", str(MODELES / "synthstrip.1.pt")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("SynthStrip a échoué : " + r.stderr[-500:])


def recaler(enfant, modele):
    # transformation affine qui envoie le modèle MNI sur le cerveau de l'enfant
    init = sitk.CenteredTransformInitializer(
        enfant, modele, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.MOMENTS)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.05, 42)       # graine fixe : résultat reproductible
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200, relaxationFactor=0.5)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(init, inPlace=False)
    return R.Execute(enfant, modele)


def vers_enfant(masque_mni, enfant, transfo):
    out = sitk.Resample(masque_mni, enfant, transfo,
                        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(out) > 0


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def traiter(ligne, modele, masque_mni, frontal_mni):
    nom = f"{ligne['site']}_{ligne['subject_id']}"
    f_brain = SORTIE / f"{nom}_brain.nii.gz"
    f_mask = SORTIE / f"{nom}_mask.nii.gz"
    f_frontal = SORTIE / f"{nom}_frontal.nii.gz"

    # 1) retrait du crâne (sauté si déjà fait)
    t0 = time.time()
    if not (f_brain.exists() and f_mask.exists()):
        # on écrit sous un nom temporaire puis on renomme : si le PC s'éteint
        # au milieu, il ne reste pas de fichier à moitié écrit sous le vrai nom
        tmp_brain = SORTIE / f"{nom}_brain.tmp.nii.gz"
        tmp_mask = SORTIE / f"{nom}_mask.tmp.nii.gz"
        synthstrip(ligne["chemin_abs"], tmp_brain, tmp_mask)
        tmp_brain.replace(f_brain)
        tmp_mask.replace(f_mask)
    t1 = time.time()

    # 2) recalage MNI -> enfant
    enfant = sitk.ReadImage(str(f_brain), sitk.sitkFloat32)
    transfo = recaler(enfant, modele)

    # 3) lobe frontal ramené dans l'espace de l'enfant, limité à son cerveau
    m_enfant = sitk.GetArrayFromImage(sitk.ReadImage(str(f_mask))) > 0.5
    m_modele = vers_enfant(masque_mni, enfant, transfo)
    frontal = vers_enfant(frontal_mni, enfant, transfo) & m_enfant

    img = sitk.GetImageFromArray(frontal.astype(np.uint8))
    img.CopyInformation(enfant)
    sitk.WriteImage(img, str(f_frontal))

    ml = np.prod(enfant.GetSpacing()) / 1000
    return {"site": ligne["site"], "subject_id": ligne["subject_id"],
            "adhd": ligne["adhd"],
            "vol_cerveau_ml": round(m_enfant.sum() * ml, 1),
            "vol_frontal_ml": round(frontal.sum() * ml, 1),
            "dice_recalage": round(dice(m_modele, m_enfant), 4),
            "duree_synthstrip_s": round(t1 - t0, 1),
            "duree_recalage_s": round(time.time() - t1, 1)}


def ajouter_ligne(fichier, ligne, colonnes):
    nouveau = not fichier.exists()
    with open(fichier, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=colonnes)
        if nouveau:
            w.writeheader()
        w.writerow(ligne)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limite", type=int, default=None,
                   help="ne traiter que les N premiers enfants restants")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # accents lisibles dans le journal sous Windows

    SORTIE.mkdir(parents=True, exist_ok=True)
    f_res, f_err = SORTIE / "resultats.csv", SORTIE / "erreurs.csv"

    df = pd.read_csv(TABLE, dtype={"subject_id": str})
    # les chemins du tableau sont relatifs au dossier notebooks/
    df["chemin_abs"] = [str((RACINE / "notebooks" / c).resolve()) for c in df["chemin"]]

    deja = set()
    if f_res.exists():
        deja = set(pd.read_csv(f_res, dtype={"subject_id": str})["subject_id"])
    # ordre mélangé (graine fixe) pour que les enfants déjà faits forment à tout
    # moment un échantillon représentatif ; Pittsburgh (que des contrôles) en dernier
    df = df.sample(frac=1, random_state=0)
    df = pd.concat([df[df["site"] != "Pittsburgh"], df[df["site"] == "Pittsburgh"]])
    a_faire = df[~df["subject_id"].isin(deja)]
    if args.limite:
        a_faire = a_faire.head(args.limite)
    print(f"{len(deja)} déjà faits, {len(a_faire)} à traiter", flush=True)

    modele = sitk.ReadImage(str(MODELES / "mni_template_brain.nii.gz"), sitk.sitkFloat32)
    masque_mni = sitk.ReadImage(str(MODELES / "mni_masque.nii.gz"), sitk.sitkUInt8)
    frontal_mni = sitk.ReadImage(str(MODELES / "frontal_MNI.nii.gz"), sitk.sitkUInt8)

    t_debut = time.time()
    for n, (_, ligne) in enumerate(a_faire.iterrows(), start=1):
        try:
            r = traiter(ligne, modele, masque_mni, frontal_mni)
            ajouter_ligne(f_res, r, COLONNES)
            etat = f"ok  dice={r['dice_recalage']:.2f} frontal={r['vol_frontal_ml']:.0f} mL"
        except Exception as e:
            # fichiers peut-être abîmés : on les efface pour tout refaire à la prochaine reprise
            nom = f"{ligne['site']}_{ligne['subject_id']}"
            for suffixe in ["brain", "mask", "frontal"]:
                (SORTIE / f"{nom}_{suffixe}.nii.gz").unlink(missing_ok=True)
            ajouter_ligne(f_err, {"site": ligne["site"], "subject_id": ligne["subject_id"],
                                  "erreur": str(e)[:300]}, ["site", "subject_id", "erreur"])
            etat = f"ÉCHEC {type(e).__name__}"
        moy = (time.time() - t_debut) / n
        reste_h = moy * (len(a_faire) - n) / 3600
        print(f"[{n}/{len(a_faire)}] {ligne['site']}_{ligne['subject_id']} {etat} "
              f"| reste ≈ {reste_h:.1f} h", flush=True)


if __name__ == "__main__":
    main()
