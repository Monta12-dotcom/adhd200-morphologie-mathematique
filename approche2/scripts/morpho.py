import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu
from skimage.measure import euler_number
from skimage.morphology import ball, skeletonize

MARGE = 8  # voxels autour du lobe frontal (pour que les boules ne touchent pas le bord)


# ---------------------------------------------------------------- outils de base
# érosion / dilatation par une boule de rayon r (mm) via la transformée de distance :
#   érosion(X, B_r)    = { x : distance(x, fond) > r }
#   dilatation(X, B_r) = { x : distance(x, X) <= r }
def edt(x):
    return ndi.distance_transform_edt(x)


def erosion(X, r):
    return edt(X) > r


def dilatation(X, r):
    return edt(~X) <= r


def ouverture(X, r):
    return dilatation(erosion(X, r), r)


def fermeture(X, r):
    return erosion(dilatation(X, r), r)


def gradient_morpho(X, r=1):
    # dilatation - érosion : une fine couche autour de la surface de l'objet
    return dilatation(X, r) & ~erosion(X, r)


# ---------------------------------------------------------------- chargement
def _iso_1mm(img, interp):
    sp = np.array(img.GetSpacing())
    if np.allclose(sp, 1.0, atol=0.02):
        return img
    taille = [int(round(n * s)) for n, s in zip(img.GetSize(), sp)]
    return sitk.Resample(img, taille, sitk.Transform(), interp, img.GetOrigin(),
                         (1.0, 1.0, 1.0), img.GetDirection(), 0, img.GetPixelID())


def charger_enfant(f_brain, f_mask, f_frontal):
    # images à 1 mm isotrope, recadrées autour du lobe frontal
    brain = _iso_1mm(sitk.ReadImage(str(f_brain), sitk.sitkFloat32), sitk.sitkLinear)
    mask = _iso_1mm(sitk.ReadImage(str(f_mask), sitk.sitkUInt8), sitk.sitkNearestNeighbor)
    front = _iso_1mm(sitk.ReadImage(str(f_frontal), sitk.sitkUInt8), sitk.sitkNearestNeighbor)
    b = sitk.GetArrayFromImage(brain)
    m = sitk.GetArrayFromImage(mask) > 0
    f = sitk.GetArrayFromImage(front) > 0
    vol_cerveau = m.sum() / 1000.0  # mL (voxels de 1 mm3)

    idx = np.argwhere(f)
    lo = np.maximum(idx.min(0) - MARGE, 0)
    hi = np.minimum(idx.max(0) + MARGE + 1, f.shape)
    s = tuple(slice(a, z) for a, z in zip(lo, hi))
    return b[s], m[s], f[s], vol_cerveau


# ---------------------------------------------------------------- étape 4
def segmenter(b, m, f):
    # Otsu 3 classes (LCR < matière grise < matière blanche), seuils propres à chaque enfant
    # et calculés seulement autour du lobe frontal
    zone = m & dilatation(f, 5)
    t1, t2 = threshold_multiotsu(b[zone], classes=3)
    lcr = m & (b <= t1)
    gm = m & (b > t1) & (b <= t2)
    wm = m & (b > t2)
    # hygiène morphologique : on retire les petits îlots isolés de matière grise
    gm_f = gm & f
    lab, n = ndi.label(gm_f, structure=np.ones((3, 3, 3)))
    if n > 0:
        tailles = np.bincount(lab.ravel())
        tailles[0] = 0
        gm_f = tailles[lab] >= 30
    return {"lcr": lcr, "gm": gm, "wm": wm, "gm_f": gm_f,
            "seuils": (float(t1), float(t2))}


# ---------------------------------------------------------------- étape 5
RAYONS_OUV = [1.0, 1.5, 2.0, 2.5, 3.0]   # mm
RAYONS_FERM = [1.0, 2.0, 3.0, 4.0]       # mm


def granulometrie_ouvertures(X, rayons=RAYONS_OUV):
    # fraction restante après chaque ouverture, et spectre (fraction partie entre deux rayons)
    v0 = X.sum()
    restes = [v0]
    e = edt(X)
    for r in rayons:
        ero = e > r
        restes.append(dilatation(ero, r).sum() if ero.any() else 0)
    restes = np.array(restes, float) / v0
    return restes, -np.diff(restes)


def granulometrie_fermetures(T, f, rayons=RAYONS_FERM):
    # la fermeture du tissu comble les sillons plus étroits que 2r : volume comblé dans le lobe
    vf = f.sum()
    d_ext = edt(~T)
    rempli = [0.0]
    for r in rayons:
        clo = edt(d_ext <= r) > r
        rempli.append((clo & ~T & f).sum() / vf)
    rempli = np.array(rempli)
    return rempli[1:], np.diff(rempli)


def squelette_epaisseur(gm):
    dt = edt(gm)
    sq = skeletonize(gm)
    e = 2 * dt[sq]
    # voisins de chaque point du squelette (26-voisinage)
    vois = ndi.convolve(sq.astype(np.uint8), np.ones((3, 3, 3), np.uint8),
                        mode="constant") - 1
    vois = vois[sq]
    n = max(sq.sum(), 1)
    return {
        "ep_moy": e.mean(), "ep_std": e.std(),
        "ep_p90": np.percentile(e, 90),
        "sq_ratio": sq.sum() / gm.sum(),
        "sq_extremites": (vois == 1).sum() / n * 1000,   # pour 1000 voxels de squelette
        "sq_jonctions": (vois >= 3).sum() / n * 1000,
    }, sq, dt


def descripteurs(b, m, f, vol_cerveau, seg=None):
    if seg is None:
        seg = segmenter(b, m, f)
    gm, wm, lcr, gm_f = seg["gm"], seg["wm"], seg["lcr"], seg["gm_f"]
    vf = f.sum()
    d = {}

    # volumes (normalisés : on ne veut pas mesurer la taille de la tête)
    d["vol_frontal_frac"] = vf / 1000 / vol_cerveau
    d["gm_frac"] = gm_f.sum() / vf
    d["wm_frac"] = (wm & f).sum() / vf
    d["lcr_frac"] = (lcr & f).sum() / vf

    # granulométrie par ouvertures de la matière grise frontale
    restes, spectre = granulometrie_ouvertures(gm_f)
    for r, v in zip(RAYONS_OUV, restes[1:]):
        d[f"ouv_reste_r{r}"] = v
    for r, v in zip(RAYONS_OUV, spectre):
        d[f"ouv_spectre_r{r}"] = v

    # granulométrie par fermetures du tissu (largeur des sillons)
    T = gm | wm
    rempli, spectre_f = granulometrie_fermetures(T, f)
    for r, v in zip(RAYONS_FERM, rempli):
        d[f"ferm_rempli_r{r}"] = v

    # squelette + transformée de distance : épaisseur corticale
    sq_stats, _, _ = squelette_epaisseur(gm_f)
    d.update(sq_stats)

    # gradient morphologique : surface et indice de plissement
    surf_gm = gradient_morpho(gm_f).sum()
    d["gm_surface_volume"] = surf_gm / gm_f.sum()
    surf_T = (gradient_morpho(T) & f).sum()
    surf_lisse = (gradient_morpho(fermeture(T, 4)) & f).sum()
    d["gyrification"] = surf_T / max(surf_lisse, 1)  # surface plissée / surface « lissée »

    # top-hat noir sur l'intensité (fermeture - image) : creux étroits (sillons fins)
    bb = b / np.median(b[wm & f])  # intensité relative à la matière blanche
    th = ndi.grey_closing(bb, footprint=ball(2)) - bb
    d["tophat_noir_moy"] = th[f & m].mean()
    d["tophat_noir_p90"] = np.percentile(th[f & m], 90)

    # topologie
    d["euler_gm"] = euler_number(gm_f, connectivity=3) / gm_f.sum() * 1000

    # contraste matière grise / blanche (marqueur de maturation, non morphologique)
    d["contraste_gm_wm"] = np.median(b[gm_f]) / np.median(b[wm & f])
    return d
