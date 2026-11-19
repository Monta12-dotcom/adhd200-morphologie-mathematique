import argparse
import json
import multiprocessing as mp
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi
from scipy.stats import kurtosis, skew
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.measure import euler_number
from skimage.morphology import ball, h_maxima, skeletonize

warnings.filterwarnings("ignore")

TARGET_MM = 1.0                      # isotropic working voxel size (mm). Cortex is 2.5-3 mm thick,
                                     # so 2 mm voxels are too coarse for any structuring element.
EXTRACT_MM = 2.0                     # brain extraction is done on a coarse 2 mm grid (fast)
OPEN_MM = 6.0                        # opening that cuts thin bridges (skull, scalp, nerves) before propagation
POLISH_MM = 10.0                     # final opening that removes skull-base leaks
GROW_MM = 2.0                        # the conservative core is grown back to the cortical surface by at most this (mm)
THR_FRACS = (0.52, 0.47, 0.42, 0.38, 0.34, 0.30)   # tissue threshold / WM level, strict to lenient
MIN_CORE_L = 1.05                    # a grown core smaller than this (L) makes the threshold search relax
MAX_HEAD_FRAC = 0.75                 # a brain mask covering more of the head than this = extraction failed
RADII_MM = [1, 2, 3, 4, 5, 6]        # structuring-element radii for (anti-)granulometry
MIN_ICV_L, MAX_ICV_L = 0.8, 2.1      # plausible brain volume window (litres)
S26 = np.ones((3, 3, 3), bool)       # 26-connectivity (8-connexity in 3D)
S6 = ndi.generate_binary_structure(3, 1)   # 6-connectivity (4-connexity in 3D)
SEG_COLORS = ["#00000000", "#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]  # none, sulcCSF, GM, WM, vent
SEG_CMAP = ListedColormap(SEG_COLORS)


# =========================================================================== operators
def B(r):
    """Ball structuring element, radius r in voxels."""
    return ball(int(r)).astype(bool)


def largest_cc(mask):
    lab, n = ndi.label(mask, S26)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def remove_small(mask, min_size=30):
    lab, n = ndi.label(mask, S26)
    if n == 0:
        return mask
    keep = np.bincount(lab.ravel()) >= min_size
    keep[0] = False
    return keep[lab]


def asf(mask, n=2):
    """Alternating sequential filter (opening then closing, growing ball)."""
    for r in range(1, n + 1):
        mask = ndi.binary_closing(ndi.binary_opening(mask, B(r)), B(r))
    return mask


def edt_erode(m, r_mm, vox):
    """Euclidean erosion by a ball of radius r_mm (distance transform > r)."""
    return ndi.distance_transform_edt(m, sampling=vox) > r_mm


def edt_dilate(m, r_mm, vox):
    return ndi.distance_transform_edt(~m, sampling=vox) <= r_mm


def edt_open(m, r_mm, vox):
    return edt_dilate(edt_erode(m, r_mm, vox), r_mm, vox) & m


def edt_close(m, r_mm, vox):
    return edt_erode(edt_dilate(m, r_mm, vox), r_mm, vox) | m


def resample_to(img, from_mm, to_mm, out_shape=None):
    """Isotropic resampling (anti-aliased when shrinking)."""
    f = from_mm / to_mm
    if f < 1:
        img = ndi.gaussian_filter(img, 0.5 * (1 / f - 1))
    if out_shape is None:
        return ndi.zoom(img, f, order=1)
    return ndi.zoom(img, np.array(out_shape) / np.array(img.shape), order=1)


DIAG = {}      # diagnostics of the last extraction (printed when a child fails)


def wm_reference(sm, head, vox, radius_mm=35.0):
    """White-matter intensity measured at the CENTRE of the head, where only brain exists
    (scalp fat can be brighter than white matter, so a global bright class is not safe).
    Centre = head centroid in x,y and the level of the largest axial head section in z.
    The WM peak is found by mean-shift (repeated median of voxels near the current estimate)."""
    area = head.sum(axis=(0, 1))
    zc = int(area.argmax())
    cx, cy, _ = ndi.center_of_mass(head)
    X, Y, Z = np.ogrid[:head.shape[0], :head.shape[1], :head.shape[2]]
    sph = head & ((((X - cx) ** 2 + (Y - cy) ** 2 + (Z - zc) ** 2) * vox[0] ** 2) < radius_mm ** 2)
    v = sm[sph]
    ref = float(np.percentile(v, 80))
    for _ in range(12):
        sel = v[(v > 0.88 * ref) & (v < 1.12 * ref)]
        if sel.size < 50:
            break
        ref = float(np.median(sel))
    g = v[(v > 0.35 * ref) & (v < 0.88 * ref)]
    gm_ref = float(np.median(g)) if g.size > 50 else 0.7 * ref
    return ref, gm_ref


def wm_seed(sm, region, vox, ref, erode_mm=6.0):
    """White-matter core: voxels near the WM intensity, eroded so thin structures vanish."""
    cand = region & (sm > 0.85 * ref) & (sm < 1.2 * ref)
    for r in (erode_mm, 5.0, 4.0, 3.0, 2.0):
        seed = largest_cc(edt_erode(cand, r, vox))
        if seed.sum() > 0:
            return seed
    return cand


def extract_brain_coarse(sm, vox):
    """Conservative brain core on a coarse grid:
    head mask -> WM and GM reference intensities from the head centre -> WM core (erosion) ->
    tissue mask (above a threshold, below fat) -> opening cuts thin bridges -> geodesic
    propagation (reconstruction) from the WM core. The tissue threshold is tried from strict
    to lenient and the STRICTEST one that grows a plausible brain is kept (strict keeps the
    scalp out; lenient is only needed when gray-matter contrast is low). Then closing, hole
    filling and a polish opening that removes skull-base leaks."""
    head = sm > 0.5 * threshold_otsu(sm)
    head = ndi.binary_fill_holes(largest_cc(ndi.binary_opening(head, B(1))))
    ref, gm_ref = wm_reference(sm, head, vox)
    seed = wm_seed(sm, head, vox, ref)
    # strict -> lenient tissue thresholds: take the strictest one that grows a plausible brain.
    # (a strict threshold keeps the scalp out; a lenient one is only needed at low GM contrast)
    for fr in THR_FRACS:
        t_tis = max(fr * ref, 0.4 * gm_ref)
        tissue = head & (sm > t_tis) & (sm < 1.4 * ref)
        op = edt_open(tissue, OPEN_MM, vox)
        br = ndi.binary_propagation(seed & op, structure=S26, mask=op) if (seed & op).any() else np.zeros_like(op)
        if br.sum() * vox[0] ** 3 / 1e6 >= MIN_CORE_L:
            break
    DIAG["t_tis_frac"] = t_tis / ref
    DIAG.update(wm_ref=ref, gm_ref=gm_ref, seed_ml=seed.sum() * vox[0] ** 3 / 1e3, head_l=head.sum() * vox[0] ** 3 / 1e6,
                tissue_l=tissue.sum() * vox[0] ** 3 / 1e6, propagated_l=br.sum() * vox[0] ** 3 / 1e6)
    br = ndi.binary_fill_holes(edt_close(br, 4, vox) & head)
    br = largest_cc(edt_open(br, POLISH_MM, vox))
    return ndi.binary_fill_holes(edt_close(br, 3, vox)), ref, gm_ref


def grow_to_surface(img, core, ref, gm_ref, vox):
    """Geodesic reconstruction of the cortical surface: starting from the conservative core,
    propagate through tissue-like voxels (above half the GM level, below fat) but never farther
    than GROW_MM from the core, so the outer gyri come back and the scalp does not."""
    sm = ndi.gaussian_filter(img, 0.7)
    d = ndi.distance_transform_edt(~core, sampling=vox)
    near = d <= GROW_MM
    t_low = float(np.clip(0.5 * gm_ref, 0.25 * ref, 0.5 * ref))
    reg = (near & (sm > t_low) & (sm < 1.4 * ref)) | core
    br = ndi.binary_propagation(core, structure=S26, mask=reg)
    br = edt_close(br, 3, vox) & near                       # brings sulcal CSF inside
    br = ndi.binary_fill_holes(br)
    br = largest_cc(edt_open(br, 2.0, vox))                 # cuts thin tendrils (meninges, nerves)
    return ndi.binary_fill_holes(br)


def brain_mask(img, vox):
    """Returns (mask at working resolution, WM reference intensity).
    Already skull-stripped -> keep the non-zero region. Still has the head -> extract."""
    DIAG.clear()
    vv = float(np.prod(vox))
    m = ndi.binary_fill_holes(largest_cc(img > 0))
    sm = ndi.gaussian_filter(resample_to(img, vox[0], EXTRACT_MM), 0.5)
    cvox = (EXTRACT_MM,) * 3
    if m.sum() * vv / 1e6 <= MAX_ICV_L:
        mc = resample_to(m.astype(np.float32), vox[0], EXTRACT_MM) > 0.5
        return m, wm_reference(sm, mc, cvox)[0]
    mc, wm_ref, gm_ref = extract_brain_coarse(sm, cvox)
    up = ndi.zoom(mc.astype(np.float32), np.array(img.shape) / np.array(mc.shape), order=1) > 0.5
    up = ndi.binary_fill_holes(up)
    DIAG["core_l"] = up.sum() * vv / 1e6
    mask = grow_to_surface(img, up, wm_ref, gm_ref, vox)
    return mask, wm_ref


def tissue_classes(img, mask, wm_ref, vox):
    """Bias correction from WM-like voxels only, then thresholds anchored to the WM peak.
    Returns corrected image, CSF/GM/WM masks (before cleaning) and the two thresholds."""
    sm = ndi.gaussian_filter(img, 0.5)
    inner = edt_erode(mask, 2.0, vox)                      # skip partial-volume boundary
    rng = inner & (sm > 0.4 * wm_ref) & (sm < 1.35 * wm_ref)
    t_gw = threshold_otsu(sm[rng])
    W = (rng & (sm > t_gw)).astype(np.float32)
    sig = 15.0 / vox[0]
    den = ndi.gaussian_filter(W, sig) + 1e-6
    bias = np.where(den > 0.02, ndi.gaussian_filter(sm * W, sig) / den, wm_ref) / wm_ref
    imgc = (sm / np.clip(bias, 0.5, 1.5)) * mask
    rng = inner & (imgc > 0.4 * wm_ref) & (imgc < 1.35 * wm_ref)
    t_gw = threshold_otsu(imgc[rng])
    gm_ref = float(np.median(imgc[rng & (imgc < t_gw)]))
    t_cg = 0.65 * gm_ref
    return imgc, t_cg, t_gw


def gran_openings(mask, radii, vox):
    """Euclidean openings of `mask` for several radii (one distance transform each)."""
    d = ndi.distance_transform_edt(mask, sampling=vox)
    return [(ndi.distance_transform_edt(~(d > r), sampling=vox) <= r) & mask for r in radii]


def gran_closing_gain(mask, radii, vox):
    """Voxels added by closings of growing radius (anti-granulometry)."""
    dc = ndi.distance_transform_edt(~mask, sampling=vox)
    return [(ndi.distance_transform_edt(dc <= r, sampling=vox) > r) & ~mask for r in radii]


def gran_curve(opened, base):
    v0 = base.sum()
    return np.array([o.sum() / v0 for o in opened]) if v0 else np.full(len(opened), np.nan)


def gran_meansize(curve):
    if np.isnan(curve).any():
        return np.nan
    spec = -np.diff(np.r_[1.0, curve])                 # granulometric spectrum
    return float((spec * np.array(RADII_MM)).sum() + curve[-1] * (RADII_MM[-1] + 1))


def ultimate_erosion(mask, vox, h_mm=1.0):
    """Ultimate erosion = regional maxima of the distance map, kept only if they rise at
    least h_mm above their surroundings (h-maxima = geodesic reconstruction). Without h,
    every flat plateau of the discrete distance map counts as an object."""
    if mask.sum() == 0:
        return np.zeros_like(mask), 0, np.nan
    d = ndi.distance_transform_edt(mask, sampling=vox)
    rm = (h_maxima(d, h_mm) > 0) & mask
    return rm, int(ndi.label(rm, S26)[1]), float(d.max())


def geodesic_diameter(mask):
    """Longest geodesic path (in voxel steps) by two geodesic-dilation sweeps."""
    if mask.sum() == 0:
        return np.nan
    mask = largest_cc(mask)
    mask = mask[ndi.find_objects(mask.astype(np.uint8))[0]]

    def sweep(seed):
        reached, last, steps = seed, seed, 0
        while True:
            new = ndi.binary_dilation(reached, structure=S26, mask=mask)
            added = new & ~reached
            if not added.any():
                return steps, last
            reached, last, steps = new, added, steps + 1

    s1 = np.zeros_like(mask)
    s1[tuple(np.argwhere(mask)[0])] = True
    _, last = sweep(s1)
    s2 = np.zeros_like(mask)
    s2[tuple(np.argwhere(last)[0])] = True
    return float(sweep(s2)[0])


def skeleton_stats(mask):
    if mask.sum() == 0:
        return np.nan, np.nan
    sk = skeletonize(mask).astype(bool)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="constant") - 1
    return float(sk.sum()), float((sk & (nb >= 3)).sum())


def stats(v, name):
    keys = ("mean", "std", "skew", "kurt", "p90")
    if v.size < 5:
        return {f"{name}_{k}": np.nan for k in keys}
    return {f"{name}_mean": float(v.mean()), f"{name}_std": float(v.std()),
            f"{name}_skew": float(skew(v)), f"{name}_kurt": float(kurtosis(v)),
            f"{name}_p90": float(np.percentile(v, 90))}


def mean_or_nan(v):
    return float(v.mean()) if v.size else np.nan


# =========================================================================== loading
def load_iso(path, target=None):
    """Load, RAS orientation, resample to isotropic `target` mm. Returns (img, original zooms)."""
    target = TARGET_MM if target is None else target
    nii = nib.as_closest_canonical(nib.load(str(path)))
    img = np.asarray(nii.get_fdata(dtype=np.float32))
    if img.ndim == 4:
        img = img[..., 0]
    z = np.array(nii.header.get_zooms()[:3], float)
    f = z / target                                     # zoom factor per axis
    if np.any(f < 1):                                  # smooth before shrinking (anti-alias)
        img = ndi.gaussian_filter(img, np.where(f < 1, 0.5 * (1 / f - 1), 0.0))
    if not np.allclose(f, 1):
        img = ndi.zoom(img, f, order=1)
    return img.astype(np.float32), z


# =========================================================================== plotting
def views(vol, c):
    """axial, coronal, sagittal slices through voxel c = (x, y, z)."""
    return [vol[:, :, c[2]].T, vol[:, c[1], :].T, vol[c[0], :, :].T]


VIEW_NAMES = ["axial", "coronal", "sagittal"]


def save_fig(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def fig_extraction(img0, mask, c, title, path):
    vmax = np.percentile(img0, 99.5)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
    for k, (a, m) in enumerate(zip(views(img0, c), views(mask, c))):
        ax[k].imshow(a, cmap="gray", origin="lower", vmin=0, vmax=vmax)
        if m.any():
            ax[k].contour(m.astype(float), [0.5], colors="red", linewidths=1.0)
        ax[k].set_title(VIEW_NAMES[k], fontsize=9)
        ax[k].axis("off")
    fig.suptitle(title + "\nred line = brain mask. It must follow the brain only "
                 "(no skull, scalp, eyes or neck).", fontsize=10, y=1.02)
    save_fig(fig, path)


def fig_segmentation(img0, seg, c, title, path):
    vmax = np.percentile(img0, 99.5)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
    for k, (a, s) in enumerate(zip(views(img0, c), views(seg, c))):
        ax[k].imshow(a, cmap="gray", origin="lower", vmin=0, vmax=vmax)
        ax[k].imshow(np.ma.masked_where(s == 0, s), cmap=SEG_CMAP, origin="lower",
                     vmin=0, vmax=4, alpha=0.55, interpolation="nearest")
        ax[k].set_title(VIEW_NAMES[k], fontsize=9)
        ax[k].axis("off")
    fig.suptitle(title + "\nblue = sulcal CSF, green = gray matter, orange = white matter, "
                 "red = ventricles", fontsize=10, y=1.02)
    save_fig(fig, path)


def fig_operators(m, zc, title, path):
    sl = lambda a: a[:, :, zc].T
    gm, sulc = m["gm"], m["sulc"]
    fig, ax = plt.subplots(2, 4, figsize=(15, 8))
    ax = ax.ravel()

    def show(i, arr, ttl, cmap="viridis", mask=None, vmin=None, vmax=None):
        a = sl(arr)
        if mask is not None:
            a = np.ma.masked_where(~sl(mask), a)
        im = ax[i].imshow(a, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax[i].set_title(ttl, fontsize=9)
        ax[i].axis("off")
        plt.colorbar(im, ax=ax[i], fraction=0.046, pad=0.02)

    show(0, m["imgc"], "bias-corrected T1\n(input of the operators)", "gray", m["mask"], 0, np.percentile(m["imgc"][m["mask"]], 99.5))
    show(1, m["edt_gm"], "distance transform of GM (mm)\n= cortical thickness proxy", "magma", gm)
    show(2, m["edt_sulc"], "distance transform of sulcal CSF (mm)\n= sulcal width proxy", "magma", sulc)
    show(3, m["bot"], "bottom-hat (closing - image)\nbright = dark narrow structures", "viridis", m["mask"])
    show(4, m["top"], "top-hat (image - opening)\nbright = small bright details", "viridis", m["mask"])
    show(5, m["grad"], "morphological gradient\n(dilation - erosion) = edges", "viridis", m["mask"])
    # ventricles: distance map + ultimate-erosion markers
    ax[6].imshow(sl(m["imgc"]), cmap="gray", origin="lower")
    vd = np.ma.masked_where(~sl(m["vent"]), sl(m["edt_vent"]))
    ax[6].imshow(vd, cmap="cool", origin="lower")
    near = m["ue"][:, :, max(zc - 1, 0):zc + 2].any(axis=2).T
    yy, xx = np.nonzero(near)
    ax[6].scatter(xx, yy, s=18, c="red", marker="x")
    ax[6].set_title(f"ventricles: distance map +\nultimate erosion markers (x), n={m['n_ue']}", fontsize=9)
    ax[6].axis("off")
    # granulometry curves
    ax[7].plot(RADII_MM, m["gm_curve"], "o-", label="GM opening", color="#22c55e")
    ax[7].plot(RADII_MM, m["wm_curve"], "o-", label="WM opening", color="#f59e0b")
    ax[7].plot(RADII_MM, m["anti"], "s--", label="closing gain (anti-granulometry)", color="#3b82f6")
    ax[7].set_xlabel("structuring element radius (mm)")
    ax[7].set_ylabel("fraction of volume")
    ax[7].set_title("granulometry: how much survives\nan opening of growing size", fontsize=9)
    ax[7].legend(fontsize=7)
    ax[7].grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    save_fig(fig, path)


def fig_granulometry(m, zc, title, path):
    """What the openings remove from the gray matter (red = removed)."""
    gm = m["gm"]
    show = [k for k, r in enumerate(RADII_MM) if r in (1, 2, 3, 4)]
    fig, ax = plt.subplots(1, len(show) + 1, figsize=(3.4 * (len(show) + 1), 3.8))
    ax[0].imshow(gm[:, :, zc].T, cmap="gray", origin="lower")
    ax[0].set_title("gray matter mask", fontsize=9)
    for a_i, k in enumerate(show, start=1):
        op = m["gm_open"][k]
        rgb = np.zeros(gm[:, :, zc].T.shape + (3,))
        rgb[op[:, :, zc].T] = (0.85, 0.85, 0.85)
        rgb[(gm & ~op)[:, :, zc].T] = (0.9, 0.1, 0.1)
        ax[a_i].imshow(rgb, origin="lower")
        ax[a_i].set_title(f"opening, radius {RADII_MM[k]:g} mm\nremoved {100 * (1 - op.sum() / gm.sum()):.0f}% of GM", fontsize=9)
    for a in ax:
        a.axis("off")
    fig.suptitle(title + "\nred = gray matter thinner than the structuring element (removed by the opening)",
                 fontsize=10, y=1.03)
    save_fig(fig, path)


# =========================================================================== one subject

# =========================================================================== pseudo-regional + texture features
OCT_AXES = ("LR", "PA", "IS")          # array axes after as_closest_canonical: x = left->right, y = post->ant, z = inf->sup


def region_features(mask, gm, sulc, edt_gm, edt_sulc, top, grad, band, cm):
    """Eight octants around the brain's own centre of mass (no atlas, no registration).
    Per octant: share of the brain, GM share, sulcal-CSF share, mean cortical thickness (EDT of GM),
    mean sulcal width (EDT of sulcal CSF), mean top-hat inside GM, mean morphological gradient on the GM/WM band.
    Then left-right asymmetry indices for each (post/ant x inf/sup) pair."""
    idx = np.indices(mask.shape, sparse=True)
    half = [idx[a] >= cm[a] for a in range(3)]
    nb = max(int(mask.sum()), 1)
    f, vals = {}, {}
    for lr, hl in (("L", False), ("R", True)):
        for ap, ha in (("P", False), ("A", True)):
            for si, hs in (("I", False), ("S", True)):
                o = mask & (half[0] == hl) & (half[1] == ha) & (half[2] == hs)
                n = max(int(o.sum()), 1)
                g, c = o & gm, o & sulc
                nm = f"oct_{lr}{ap}{si}"
                v = {"vol_frac": o.sum() / nb, "gm_frac": g.sum() / n, "sulc_frac": c.sum() / n,
                     "gmthick": mean_or_nan(edt_gm[g]), "sulcwidth": mean_or_nan(edt_sulc[c]),
                     "gm_tophat": mean_or_nan(top[g]), "band_grad": mean_or_nan(grad[o & band])}
                for k, x in v.items():
                    f[f"{nm}_{k}"] = float(x)
                    vals[(lr, ap, si, k)] = float(x)
    for ap in "PA":
        for si in "IS":
            for k in ("vol_frac", "gm_frac", "sulc_frac", "gmthick"):
                l, r = vals[("L", ap, si, k)], vals[("R", ap, si, k)]
                f[f"asym_{ap}{si}_{k}"] = (l - r) / (l + r) if np.isfinite(l) and np.isfinite(r) and (l + r) else np.nan
    return f


def texture_features(imgm, gm, wm):
    """Intensity texture inside the grey matter, from grey-level morphology (grey openings/closings of
    increasing size, in the spirit of a grey-level granulometry). imgm is normalised to the WM median."""
    f = {}
    v = imgm[gm]
    f.update({"tex_gm_int_mean": float(v.mean()) if v.size else np.nan,
              "tex_gm_int_std": float(v.std()) if v.size else np.nan,
              "tex_gm_int_skew": float(skew(v)) if v.size > 5 else np.nan,
              "tex_gm_int_kurt": float(kurtosis(v)) if v.size > 5 else np.nan,
              "tex_wm_int_std": float(imgm[wm].std()) if wm.any() else np.nan})
    for r in (2, 3):
        fp = B(r)
        op = ndi.grey_opening(imgm, footprint=fp)
        cl = ndi.grey_closing(imgm, footprint=fp)
        f[f"tex_gm_open_r{r}"] = float((imgm - op)[gm].mean()) if gm.any() else np.nan
        f[f"tex_gm_close_r{r}"] = float((cl - imgm)[gm].mean()) if gm.any() else np.nan
    f["tex_gm_open_ratio"] = f["tex_gm_open_r3"] / f["tex_gm_open_r2"] if f["tex_gm_open_r2"] else np.nan
    return f


def bbox_slices(mask, margin=6):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + 1 + margin, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def process_subject(path, meta, out_dir, make_figs=True):
    t0 = time.time()
    pid, label, site = str(meta["participant_id"]), str(meta["label"]), str(meta["site"])
    sd = out_dir / f"{pid}_{label}_{site}"
    head = f"{pid} | {label} | {site}"
    vox = (TARGET_MM,) * 3
    vvox = TARGET_MM ** 3

    # ---- 1. load + brain extraction (on the whole head), picture first, check afterwards
    img_full, zooms = load_iso(path)
    mask_full, wm_ref = brain_mask(img_full, vox)
    icv_l = float(mask_full.sum() * vvox / 1e6)
    cfull = [int(round(x)) for x in (ndi.center_of_mass(mask_full) if mask_full.any()
                                     else tuple(s // 2 for s in img_full.shape))]
    whole_head = DIAG.get("head_l") is not None and icv_l / DIAG["head_l"] > MAX_HEAD_FRAC
    plausible = MIN_ICV_L <= icv_l <= MAX_ICV_L and not whole_head
    if make_figs or not plausible:               # a failed child always gets its picture
        fig_extraction(img_full, mask_full, cfull, f"{head} | brain volume {icv_l:.2f} L | original voxels "
                       f"{zooms[0]:.2f}x{zooms[1]:.2f}x{zooms[2]:.2f} mm", sd / "01_brain_extraction.png")
    if not plausible:
        DIAG["mask_l"] = icv_l
        dg = ", ".join(f"{k}={v:.2f}" for k, v in DIAG.items())
        raise ValueError(f"brain mask volume {icv_l:.2f} L implausible (expected {MIN_ICV_L}-{MAX_ICV_L} L"
                         f"{', and it covers the whole head' if whole_head else ''}): skull stripping failed [{dg}]")

    # ---- 2. crop to the brain (faster) and segment
    sl = bbox_slices(mask_full)
    img0, mask = img_full[sl], mask_full[sl]
    imgc, t_cg, t_gw = tissue_classes(img0, mask, wm_ref, vox)
    wm_raw = mask & (imgc >= t_gw)
    wm = remove_small(ndi.binary_opening(wm_raw, B(1)), 500)         # no opening on GM: cortex is thin
    gm = remove_small(mask & (imgc >= t_cg) & (imgc < t_gw) & ~wm, 500)
    csf = remove_small(mask & (imgc < t_cg) & edt_erode(mask, 1.5, vox), 100)
    tissue = gm | wm
    cm = ndi.center_of_mass(mask)
    # ventricles: opening removes thin sulcal CSF, geodesic reconstruction from a central marker
    csf_core = edt_open(csf, 2.0, vox)
    zz, yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
    sphere = (((zz - cm[0]) ** 2 + (yy - cm[1]) ** 2 + (xx - cm[2]) ** 2) * TARGET_MM ** 2) < 30.0 ** 2
    marker = csf_core & sphere
    vent = ndi.binary_propagation(marker, structure=S6, mask=csf_core) if marker.any() \
        else np.zeros_like(mask)
    vent = edt_dilate(vent, 2.5, vox) & csf if vent.any() else vent
    vent_ml = float(vent.sum() * vvox / 1e3)
    vent_ok = 2.0 <= vent_ml <= 150.0
    if not vent_ok:
        vent = np.zeros_like(mask)
    sulc = csf & ~vent
    seg = np.zeros(mask.shape, np.uint8)
    seg[sulc], seg[gm], seg[wm], seg[vent] = 1, 2, 3, 4
    c = [int(round(x)) for x in cm]
    zc = int(round(ndi.center_of_mass(vent)[2])) if vent_ok else c[2] + int(8 / TARGET_MM)
    zc = int(np.clip(zc, 0, mask.shape[2] - 1))
    cz = [c[0], c[1], zc]
    if make_figs:
        fig_segmentation(img0, seg, cz, f"{head} | tissue segmentation (ventricles {'ok' if vent_ok else 'NOT FOUND'}, "
                         f"{vent_ml:.0f} mL)", sd / "02_segmentation.png")

    # ---- 3. operator maps
    wm_med = np.median(imgc[wm]) if wm.any() else 1.0
    imgm = (imgc / wm_med) * mask
    m = dict(mask=mask, gm=gm, wm=wm, sulc=sulc, vent=vent, imgc=imgc)
    m["edt_gm"] = ndi.distance_transform_edt(gm, sampling=vox)
    edt_wm = ndi.distance_transform_edt(wm, sampling=vox)
    m["edt_sulc"] = ndi.distance_transform_edt(sulc, sampling=vox)
    m["edt_vent"] = ndi.distance_transform_edt(vent, sampling=vox)
    m["bot"] = ndi.grey_closing(imgm, footprint=B(1)) - imgm
    m["top"] = imgm - ndi.grey_opening(imgm, footprint=B(1))
    m["grad"] = ndi.grey_dilation(imgm, footprint=B(1)) - ndi.grey_erosion(imgm, footprint=B(1))
    band = ndi.binary_dilation(gm, B(1)) & ndi.binary_dilation(wm, B(1))
    m["gm_open"] = gran_openings(gm, RADII_MM, vox)
    wm_open = gran_openings(wm, RADII_MM, vox)
    gains = gran_closing_gain(tissue, RADII_MM, vox)
    m["gm_curve"], m["wm_curve"] = gran_curve(m["gm_open"], gm), gran_curve(wm_open, wm)
    m["anti"] = np.array([g.sum() / max(tissue.sum(), 1) for g in gains])
    m["ue"], m["n_ue"], max_d = ultimate_erosion(vent, vox)
    if make_figs:
        fig_operators(m, zc, f"{head} | morphological operators (axial slice through the ventricles)",
                      sd / "03_operators.png")
        fig_granulometry(m, zc, f"{head} | granulometry", sd / "04_granulometry.png")

    # ---- 4. global features
    nb = max(int(mask.sum()), 1)
    f = {"participant_id": pid, "brain_volume_L": icv_l, "ICV": float(mask_full.sum() * vvox),
         "gm_frac": gm.sum() / nb, "wm_frac": wm.sum() / nb, "sulc_frac": sulc.sum() / nb,
         "vent_frac": vent.sum() / nb, "vent_ok": int(vent_ok), "thr_csf_gm": t_cg / wm_ref,
         "thr_gm_wm": t_gw / wm_ref}
    f.update(stats(m["edt_gm"][gm], "gmthick"))
    f.update({"wmthick_mean": mean_or_nan(edt_wm[wm]),
              "sulcwidth_mean": mean_or_nan(m["edt_sulc"][sulc]),
              "bothat_mean": mean_or_nan(m["bot"][mask]),
              "bothat_energy": float((m["bot"][mask] ** 2).mean()),
              "tophat_mean": mean_or_nan(m["top"][mask]),
              "grad_mean": mean_or_nan(m["grad"][band]),
              "grad_std": float(m["grad"][band].std()) if band.any() else np.nan,
              "gm_gran_meansize": gran_meansize(m["gm_curve"]),
              "wm_gran_meansize": gran_meansize(m["wm_curve"])})
    for i, r in enumerate(RADII_MM):
        f[f"gm_gran_curve_{r}mm"] = m["gm_curve"][i]
        f[f"wm_gran_curve_{r}mm"] = m["wm_curve"][i]
        f[f"antigran_{r}mm"] = m["anti"][i]
    for nm, mk in (("gm", gm), ("wm", wm), ("vent", vent)):
        f[f"{nm}_euler"] = int(euler_number(mk, connectivity=1)) if mk.any() else np.nan
        f[f"{nm}_ncc"] = int(ndi.label(mk, S26)[1])
    wm2 = resample_to(wm.astype(np.float32), TARGET_MM, 2.0) > 0.5       # skeleton on a coarse copy (fast)
    sk_len, sk_bp = skeleton_stats(wm2)
    f["wm_skel_len_norm"] = sk_len / max(wm2.sum(), 1)
    f["wm_skel_branch_norm"] = sk_bp / max(wm2.sum(), 1)
    f["vent_ue_n"], f["vent_maxdist_mm"] = m["n_ue"], max_d
    f["vent_geo_diam_steps"] = geodesic_diameter(vent)
    f["vent_volume_mm3"] = float(vent.sum() * vvox)
    # ---- 4b. pseudo-regional (octant) and texture features
    f.update(region_features(mask, gm, sulc, m["edt_gm"], m["edt_sulc"], m["top"], m["grad"], band, cm))
    f.update(texture_features(imgm, gm, wm))
    # keep the segmentation so that more features can be computed later WITHOUT reprocessing the images
    seg_dir = out_dir / "seg"
    seg_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(seg_dir / f"{pid}.npz", seg=seg, start=np.array([x.start for x in sl]),
                        shape=np.array(mask_full.shape), wm_median=wm_med, zooms=np.array(zooms))

    # thumbnails for the overview sheets (axial slice through the ventricles, full-head coordinates)
    thumb = None
    if make_figs:
        zf = zc + sl[2].start
        seg_full = np.zeros(mask_full.shape, np.uint8)
        seg_full[sl] = seg
        thumb = {"t1": img_full[:, :, zf].T, "mask": mask_full[:, :, zf].T, "seg": seg_full[:, :, zf].T}
    curves = {"gm": m["gm_curve"], "wm": m["wm_curve"], "anti": m["anti"]}
    return dict(features=f, thumb=thumb, curves=curves, icv_l=icv_l, zooms=zooms,
                seconds=time.time() - t0)


# =========================================================================== overview figures
def label_colors(labels):
    uniq = sorted(set(labels))
    pal = plt.get_cmap("tab10")
    return {l: pal(i) for i, l in enumerate(uniq)}


def contact_sheet(items, kind, path, title):
    n = len(items)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.5 * nrow))
    ax = np.atleast_1d(ax).ravel()
    for a in ax:
        a.axis("off")
    for a, it in zip(ax, items):
        t = it["thumb"]
        vmax = np.percentile(t["t1"], 99.5)
        a.imshow(t["t1"], cmap="gray", origin="lower", vmin=0, vmax=vmax)
        if kind == "mask":
            if t["mask"].any():
                a.contour(t["mask"].astype(float), [0.5], colors="red", linewidths=0.9)
        else:
            a.imshow(np.ma.masked_where(t["seg"] == 0, t["seg"]), cmap=SEG_CMAP, origin="lower",
                     vmin=0, vmax=4, alpha=0.55, interpolation="nearest")
        a.set_title(f"{it['pid']} | {it['label']} | {it['site']}\n{it['icv_l']:.2f} L", fontsize=8)
    fig.suptitle(title, fontsize=12, y=1.03)
    save_fig(fig, path)


def granulometry_overview(items, path):
    cols = label_colors([it["label"] for it in items])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for it in items:
        for a, key in zip(ax, ("gm", "wm", "anti")):
            a.plot(RADII_MM, it["curves"][key], "-", color=cols[it["label"]], alpha=0.7, lw=1.2)
    for a, t in zip(ax, ("GM: fraction surviving opening", "WM: fraction surviving opening",
                         "tissue added by closing (anti-granulometry)")):
        a.set_title(t, fontsize=10)
        a.set_xlabel("structuring element radius (mm)")
        a.grid(alpha=0.3)
    for l, col in cols.items():
        ax[0].plot([], [], color=col, label=f"label {l}")
    ax[0].legend(fontsize=8)
    fig.suptitle("Granulometry curves, one line per child (a first look, NOT a statistical test)", fontsize=11, y=1.03)
    save_fig(fig, path)


# =========================================================================== data access
def index_t1(data_dir):
    """Map participant id -> T1 file (BIDS: sub-<id>/.../anat/*T1w.nii.gz)."""
    for pat in ("**/anat/*T1w.nii*", "**/*T1w*.nii*", "**/*.nii.gz"):
        files = sorted(data_dir.glob(pat))
        if files:
            break
    idx = {}
    for p in files:
        mt = re.search(r"sub-([A-Za-z0-9]+)", str(p.relative_to(data_dir)))
        if mt:
            idx.setdefault(mt.group(1), []).append(p)
    return idx


def find_t1(idx, pid):
    for k in (pid, pid.lstrip("0")):
        for key, v in idx.items():
            if key == k or key.lstrip("0") == k.lstrip("0"):
                return v[0], len(v)
    return None, 0


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _work(job):
    """Runs in a worker process. Returns a small dict (never raises)."""
    global TARGET_MM, OPEN_MM, POLISH_MM, GROW_MM
    k, path, row, out_dir, make_figs, params, n_files = job
    TARGET_MM, OPEN_MM, POLISH_MM, GROW_MM = params     # spawned workers re-import the module: pass the settings
    qc = {"participant_id": str(row["participant_id"]), "site": row["site"], "label": row["label"], "t1_file": path}
    try:
        r = process_subject(path, row, Path(out_dir), make_figs)
        qc.update(status="ok", brain_volume_L=round(r["icv_l"], 3), seconds=round(r["seconds"], 1),
                  orig_voxel_mm="x".join(f"{z:.2f}" for z in r["zooms"]), n_t1_files=n_files)
        curves = {k_: np.asarray(v, float).tolist() for k_, v in r["curves"].items()}
        return dict(k=k, ok=True, qc=qc, row={**r["features"], **{c: v for c, v in row.items() if c != "participant_id"}},
                    curves=curves, thumb=r["thumb"], icv_l=r["icv_l"])
    except Exception as e:
        qc["status"] = f"FAILED: {type(e).__name__}: {e}"
        return dict(k=k, ok=False, qc=qc)


def main():
    global TARGET_MM, OPEN_MM, POLISH_MM, GROW_MM
    root_default = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(root_default))
    ap.add_argument("--subset", default=None, help="default: <root>/reports/subset_main.csv")
    ap.add_argument("--data_dir", default=None, help="default: <root>/ADHD200_BIDS")
    ap.add_argument("--n", type=int, default=20, help="number of children (first rows of the subset)")
    ap.add_argument("--all", action="store_true", help="process every child of the subset")
    ap.add_argument("--balanced", action="store_true", help="take n/2 children per label instead of the first n rows")
    ap.add_argument("--n_jobs", type=int, default=1, help="children processed in parallel (each needs ~2-3 GB of RAM)")
    ap.add_argument("--figs_n", type=int, default=20, help="only the first N children get pictures (faster)")
    ap.add_argument("--out", default=None, help="default: <root>/reports/pipeline_first<n> or pipeline_all")
    ap.add_argument("--target_mm", type=float, default=TARGET_MM, help="working voxel size; 1.0 recommended, 1.5 = faster")
    ap.add_argument("--open_mm", type=float, default=OPEN_MM, help="brain extraction: opening radius that cuts bridges")
    ap.add_argument("--polish_mm", type=float, default=POLISH_MM, help="brain extraction: final opening radius")
    ap.add_argument("--grow_mm", type=float, default=GROW_MM, help="brain extraction: max growth of the core toward the cortical surface")
    a = ap.parse_args()
    TARGET_MM, OPEN_MM, POLISH_MM, GROW_MM = a.target_mm, a.open_mm, a.polish_mm, a.grow_mm
    params = (TARGET_MM, OPEN_MM, POLISH_MM, GROW_MM)

    root = Path(a.root)
    subset = pd.read_csv(a.subset or root / "reports" / "subset_main.csv", dtype={"participant_id": str})
    data_dir = Path(a.data_dir) if a.data_dir else root / "ADHD200_BIDS"
    tag = "all" if a.all else f"first{a.n}"
    out = Path(a.out) if a.out else root / "reports" / f"pipeline_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    if a.all:
        sel = subset
    elif a.balanced:
        sel = subset.groupby("label", group_keys=False).head(a.n // subset["label"].nunique())
    else:
        sel = subset.head(a.n)
    sel = sel.reset_index(drop=True)
    print(f"{len(sel)} children selected ({'all' if a.all else 'balanced by label' if a.balanced else 'first rows of the file'}):")
    print(sel.groupby(["site", "label"]).size().unstack(fill_value=0).to_string(), "\n")

    idx = index_t1(data_dir)
    print(f"(index of the T1 files in {data_dir}: {sum(len(v) for v in idx.values())} files, "
          f"only used to locate the {len(sel)} selected children; nothing else is processed)\n")

    # ---- resume: children already finished in an earlier run are kept
    part_path = out / "partial.jsonl"
    done = {}
    if part_path.exists():
        for line in part_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done[str(r["qc"]["participant_id"])] = r
            except Exception:
                pass
        print(f"resuming: {len(done)} children already done in {part_path.name}\n")
    order = {str(p): i for i, p in enumerate(sel["participant_id"])}

    jobs, qc_extra = [], {}
    for k, row in sel.iterrows():
        pid = str(row["participant_id"])
        if pid in done:
            continue
        path, n_files = find_t1(idx, pid)
        if path is None:
            qc_extra[pid] = {"participant_id": pid, "site": row["site"], "label": row["label"], "t1_file": "",
                             "status": "NO T1 FILE FOUND"}
            print(f"{pid}: no T1 file found")
            continue
        jobs.append((k, str(path), row.to_dict(), str(out), k < a.figs_n, params, n_files))
    print(f"{len(jobs)} children to process with {a.n_jobs} worker(s); pictures for the first {a.figs_n}\n")

    items, fails = [], {}
    t0, finished = time.time(), 0

    def handle(res):
        nonlocal finished
        finished += 1
        q = res["qc"]
        el = time.time() - t0
        eta = el / finished * (len(jobs) - finished) / 60
        if res["ok"]:
            with open(part_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"qc": q, "row": res["row"], "curves": res["curves"]},
                                    default=_json_default) + "\n")
            done[q["participant_id"]] = {"qc": q, "row": res["row"], "curves": res["curves"]}
            if res["thumb"] is not None:
                items.append(dict(pid=q["participant_id"], label=str(q["label"]), site=str(q["site"]),
                                  thumb=res["thumb"], icv_l=res["icv_l"], k=res["k"]))
            print(f"[{finished}/{len(jobs)}] {q['participant_id']} ok  brain {res['icv_l']:.2f} L  "
                  f"{q['seconds']:.0f}s | {el / 60:.1f} min elapsed, ~{eta:.0f} min left", flush=True)
        else:
            fails[q["participant_id"]] = q
            print(f"[{finished}/{len(jobs)}] {q['participant_id']} {q['status'][:230]}", flush=True)

    if a.n_jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=a.n_jobs, mp_context=mp.get_context("spawn")) as ex:
            for fu in as_completed([ex.submit(_work, j) for j in jobs]):
                handle(fu.result())
    else:
        for j in jobs:
            handle(_work(j))

    # ---- final tables
    ok_rows = sorted(done.values(), key=lambda r: order.get(str(r["qc"]["participant_id"]), 1e9))
    feats = pd.DataFrame([r["row"] for r in ok_rows])
    feats.to_csv(out / f"features_{tag}.csv", index=False)
    qc_all = [r["qc"] for r in ok_rows] + list(fails.values()) + list(qc_extra.values())
    qc_all.sort(key=lambda q: order.get(str(q["participant_id"]), 1e9))
    pd.DataFrame(qc_all).to_csv(out / "qc_summary.csv", index=False)

    if items:
        items.sort(key=lambda it: it["k"])
        contact_sheet(items, "mask", out / "overview_brain_extraction.png",
                      "Brain extraction, axial slice through the ventricles (red = brain mask)")
        contact_sheet(items, "seg", out / "overview_segmentation.png",
                      "Tissue segmentation (blue sulcal CSF, green GM, orange WM, red ventricles)")
    curve_items = [dict(label=str(r["qc"]["label"]), curves=r["curves"]) for r in ok_rows]
    if curve_items:
        granulometry_overview(curve_items, out / "overview_granulometry.png")

    n_ok = len(ok_rows)
    print(f"\nDone: {n_ok}/{len(sel)} children processed, {len(fails)} failed, {len(qc_extra)} without a T1 file.")
    if n_ok:
        v = feats["brain_volume_L"]
        print(f"Brain volume: min {v.min():.2f}  median {v.median():.2f}  max {v.max():.2f} L")
        print(feats.groupby(["site", "label"]).size().unstack(fill_value=0).to_string())
    for q in fails.values():
        print(f"  FAILED {q['participant_id']} ({q['site']}): {q['status'][:160]}")
    print(f"Everything is in:\n  {out}\nMain output: features_{tag}.csv (features + metadata), qc_summary.csv")


if __name__ == "__main__":
    mp.freeze_support()
    main()