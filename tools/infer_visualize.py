"""Run baseline + PFESA inference on valid CBCT volumes and save slice overlay PNGs.
Usage (from repo root):
  python tools/infer_visualize.py --data /path/NC-release-data-checked --base best_TMamba3D.pth \
      --pfesa best_TMambaPFESA3D.pth --out out --n 4
Colours in prediction panels: green = correct tooth (TP), red = false positive, blue = missed tooth (FN)."""
import os, sys, glob, argparse, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import inference as I
from lib import utils, models, testers
import lib.transforms.three as T


def build(name, ckpt, gpu="0"):
    p = dict(I.params_3D_CBCT_Tooth)
    p.update(model_name=name, pretrain=ckpt, dimension="3d", scaling_version="TINY", save_dir=None, is_visual=False,
             dataset_name="3D-CBCT-Tooth",
             device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # IMPORTANT: TMamba3D keeps SpectralGatingBlocks / positional embeds / GateModules in plain python lists,
    # so they are NOT in the checkpoint; they are re-created from the RNG. test.py seeds right before get_model,
    # so we must do the same to reproduce the trained model exactly.
    utils.reproducibility(p["seed"], p["deterministic"], p["benchmark"])
    model = models.get_model(p)
    tester = testers.get_tester(p, model)
    tester.load()
    model.eval()
    return p, tester


def dice(a, b):
    a, b = a > 0, b > 0
    return 2.0 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def overlay(img, gt, pred=None):
    g = np.clip((img - img.min()) / (np.ptp(img) + 1e-8), 0, 1)
    rgb = np.stack([g, g, g], -1)
    if pred is None:  # ground truth panel: green tooth
        m = gt > 0
        rgb[m] = 0.5 * rgb[m] + 0.5 * np.array([0, 1, 0])
        return rgb
    tp, fp, fn = (gt > 0) & (pred > 0), (gt == 0) & (pred > 0), (gt > 0) & (pred == 0)
    for m, c in [(tp, [0, 1, 0]), (fp, [1, 0, 0]), (fn, [0, 0.4, 1])]:
        rgb[m] = 0.4 * rgb[m] + 0.6 * np.array(c)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder that contains valid/images and valid/labels")
    ap.add_argument("--base", required=True); ap.add_argument("--pfesa", required=True)
    ap.add_argument("--out", default="viz_out"); ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--names", nargs="*", default=None, help="optional file names (without path) to use")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(a.data, "valid", "images", "*.nii.gz")))
    if a.names:
        imgs = [p for p in imgs if os.path.basename(p) in a.names]
    imgs = imgs[:a.n]
    pb, tb = build("TMamba3D", a.base)
    pp, tp_ = build("TMambaPFESA3D", a.pfesa)
    tf = T.ComposeTransforms([T.ClipAndShift(pb["clip_lower_bound"], pb["clip_upper_bound"]),
                              T.ToTensor(pb["clip_lower_bound"], pb["clip_upper_bound"]),
                              T.Normalize(pb["normalize_mean"], pb["normalize_std"])])
    rows = []
    for ip in imgs:
        name = os.path.basename(ip).split(".")[0]
        lp = os.path.join(a.data, "valid", "labels", os.path.basename(ip))
        raw, _ = utils.load_image_or_label(ip, pb["resample_spacing"], type="image")
        gt, _ = utils.load_image_or_label(lp, pb["resample_spacing"], type="label")
        x, _ = tf(raw, None)
        x = torch.FloatTensor(x.numpy()).unsqueeze(0).unsqueeze(0).to(pb["device"])
        with torch.no_grad():
            yb = torch.argmax(tb.split_test(x), 1).squeeze(0).cpu().numpy()
            yp = torch.argmax(tp_.split_test(x), 1).squeeze(0).cpu().numpy()
        db, dp = dice(gt, yb), dice(gt, yp)
        rows.append((name, db, dp)); print(name, "DSC baseline %.4f | PFESA %.4f" % (db, dp), flush=True)
        np.save(os.path.join(a.out, name + "_pred_base.npy"), yb.astype(np.uint8))
        np.save(os.path.join(a.out, name + "_pred_pfesa.npy"), yp.astype(np.uint8))
        zs = np.where((gt > 0).sum((0, 1)) > 0)[0]
        picks = [zs[int(len(zs) * q)] for q in (0.25, 0.5, 0.75)]
        fig, ax = plt.subplots(3, 4, figsize=(14, 10.5))
        for r, k in enumerate(picks):
            sl = lambda v: v[:, :, k].T
            g = sl(raw); g = np.clip((g - g.min()) / (np.ptp(g) + 1e-8), 0, 1)
            panels = [(np.stack([g, g, g], -1), "CBCT slice"),
                      (overlay(sl(raw), sl(gt)), "Ground truth"),
                      (overlay(sl(raw), sl(gt), sl(yb)), "T-Mamba3D (DSC %.4f)" % db),
                      (overlay(sl(raw), sl(gt), sl(yp)), "+PFESA (DSC %.4f)" % dp)]
            for c, (im, t) in enumerate(panels):
                ax[r, c].imshow(im, origin="lower"); ax[r, c].axis("off")
                if r == 0: ax[r, c].set_title(t)
                else: ax[r, c].set_title("slice %d" % k if c == 0 else "", fontsize=9)
        fig.suptitle("%s   (green=correct, red=false positive, blue=missed)" % name)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, name + "_compare.png"), dpi=130); plt.close(fig)
    with open(os.path.join(a.out, "per_volume_dsc.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["volume", "dsc_baseline", "dsc_pfesa"]); w.writerows(rows)
    print("saved to", a.out)


if __name__ == "__main__":
    main()
