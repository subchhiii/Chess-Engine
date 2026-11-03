"""Free improvements on the TEST set, no retraining:
 - test-time augmentation (average the prediction for a position and its colour-flipped twin)
 - ensembles (average several models' predictions)"""
import os, glob, numpy as np
from analyze_models import predict, metrics, DATA_DIR

def flip_np(X):
    """Colour-flip: swap colours, mirror ranks (same position seen from the other side)."""
    X = X.astype(np.float32)
    pieces = np.concatenate([X[:, 6:12], X[:, 0:6]], 1)[:, :, ::-1]
    stm = 1 - X[:, 12:13]
    castle = np.concatenate([X[:, 15:17], X[:, 13:15]], 1)
    ep = X[:, 17:18][:, :, ::-1]
    return np.ascontiguousarray(np.concatenate([pieces, stm, castle, ep], 1))

z = np.load(glob.glob(os.path.join(DATA_DIR, "*test_cnn.npz"))[0])
X, y = z["X"], z["y"].astype(np.float64)
Xf = flip_np(X)

def has(name): return os.path.exists(os.path.join(DATA_DIR, name + ".pt"))
def tta(name):
    return (predict(name, X).astype(np.float64) + predict(name, Xf).astype(np.float64)) / 2
def single(name): return predict(name, X).astype(np.float64)

mlp, cnn_v2, cnn_v2t = single("mlp"), single("cnn_v2"), tta("cnn_v2")
combos = {
    "CNN v2 (baseline)":          cnn_v2,
    "CNN v2 + flip TTA":          cnn_v2t,
    "CNN v2 + MLP":               (cnn_v2 + mlp) / 2,
    "CNN v2 (TTA) + MLP":         (cnn_v2t + mlp) / 2,
}
if has("cnn_v3"):
    cnn_v3, cnn_v3t = single("cnn_v3"), tta("cnn_v3")
    combos.update({
        "CNN v3":                         cnn_v3,
        "CNN v3 + flip TTA":              cnn_v3t,
        "CNN v3 (TTA) + MLP":             (cnn_v3t + mlp) / 2,
        "CNN v2 + v3 (both TTA)":         (cnn_v2t + cnn_v3t) / 2,
        "CNN v2 + v3 (TTA) + MLP":        (cnn_v2t + cnn_v3t + mlp) / 3,
        "CNN v2 + v3 (TTA) + 0.5 MLP":    (cnn_v2t + cnn_v3t + 0.5 * mlp) / 2.5,
    })
    extra_tta = {}
    for extra in ("cnn_v5", "cnn_v6", "cnn_v7", "hyb_b", "hyb_d", "gph_b", "gph_d"):                 # fine-tuned / longer-trained models (if you trained them)
        if has(extra):
            pe, pet = single(extra), tta(extra)
            extra_tta[extra] = pet
            tag = extra.replace("cnn_", "").replace("v", "v")
            combos.update({
                f"CNN {tag}":                     pe,
                f"CNN {tag} + flip TTA":          pet,
                f"CNN v3 + {tag} (both TTA)":     (cnn_v3t + pet) / 2,
                f"CNN v2+v3+{tag} (TTA) + MLP":   (cnn_v2t + cnn_v3t + pet + mlp) / 4,
            })
    if has("cnn_v4"):                                  # the model trained with the extra Lichess data
        cnn_v4, cnn_v4t = single("cnn_v4"), tta("cnn_v4")
        combos.update({
            "CNN v4 (Lichess)":               cnn_v4,
            "CNN v4 + flip TTA":              cnn_v4t,
            "CNN v3 + v4 (both TTA)":         (cnn_v3t + cnn_v4t) / 2,
            "CNN v2+v3+v4 (TTA) + MLP":       (cnn_v2t + cnn_v3t + cnn_v4t + mlp) / 4,
        })
    for hy in ("hyb_b", "hyb_d"):                                # hybrid models: a different architecture, so a useful partner
        if hy in extra_tta and "cnn_v7" in extra_tta:
            combos[f"{hy} + CNN v7 (both TTA)"] = (extra_tta["cnn_v7"] + extra_tta[hy]) / 2
            combos[f"{hy} x2 + CNN v7 (TTA)"] = (extra_tta["cnn_v7"] + 2 * extra_tta[hy]) / 3
    if "hyb_b" in extra_tta and "hyb_d" in extra_tta:
        combos["hyb_b + hyb_d (both TTA)"] = (extra_tta["hyb_b"] + extra_tta["hyb_d"]) / 2
        if "cnn_v7" in extra_tta:
            combos["hyb_b + hyb_d + CNN v7 (TTA)"] = (extra_tta["hyb_b"] + extra_tta["hyb_d"] + extra_tta["cnn_v7"]) / 3
    for g in [n for n in extra_tta if n.startswith("gph")]:                 # graph model on its own and combined with the hybrids
        for hy in [n for n in extra_tta if n.startswith("hyb")]:
            combos[f"{hy} + {g} (both TTA)"] = (extra_tta[hy] + extra_tta[g]) / 2
    if "cnn_v6" in extra_tta and "cnn_v7" in extra_tta:         # the two fine-tuned models together
        f6, f7 = extra_tta["cnn_v6"], extra_tta["cnn_v7"]
        combos["CNN v6 + v7 (both TTA)"] = (f6 + f7) / 2
        combos["CNN v6 + v7 + v3 (TTA)"] = (f6 + f7 + cnn_v3t) / 3
        combos["CNN v6 + v7 + v3 (TTA) + MLP"] = (f6 + f7 + cnn_v3t + mlp) / 4
        if has("cnn_v4"):
            combos["CNN v6 + v7 + v4 (TTA)"] = (f6 + f7 + cnn_v4t) / 3
            combos["CNN v6+v7+v3+v4 (TTA)"] = (f6 + f7 + cnn_v3t + cnn_v4t) / 4
            combos["CNN v6+v7+v3+v4 (TTA) + MLP"] = (f6 + f7 + cnn_v3t + cnn_v4t + mlp) / 5
lines = [f"{'combination':<32}{'MAE':>8}{'MSE':>8}{'RMSE':>8}{'R2':>8}{'corr':>8}{'sign':>8}"]
for k, v in combos.items():
    r = metrics(v, y)
    lines.append(f"{k:<32}{r['MAE']:>8.4f}{r['MSE']:>8.4f}{r['RMSE']:>8.4f}{r['R2']:>8.3f}{r['corr']:>8.3f}{r['sign_agree']:>8.3f}")
print("\n".join(lines))
open(os.path.join(DATA_DIR, "quick_wins_results.txt"), "w").write("\n".join(lines) + "\n")
