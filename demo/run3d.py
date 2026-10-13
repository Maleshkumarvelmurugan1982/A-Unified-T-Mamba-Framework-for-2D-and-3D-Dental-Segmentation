# run3d.py  -- 3D T-Mamba (baseline vs PFESA) demo, runs INSIDE your existing Kaggle session
# (uses the conda env 'tmamba' you already built at /kaggle/working/miniconda/envs/tmamba)
import os, re, glob, subprocess, shutil, sys, time
ENV = "/kaggle/working/miniconda/envs/tmamba"; PY = ENV + "/bin/python"
assert os.path.exists(PY), "tmamba env not found - run this in the SAME session where you built the env"
N_VOLUMES = globals().get("N_VOLUMES", 3)
SLICE = "auto"
OUT = "/kaggle/working/viz3d"

def sh(cmd, keep=None):
    env = dict(os.environ); env.pop("PYTHONPATH", None)
    os.makedirs("/kaggle/working/libcuda", exist_ok=True)
    cu = (glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so*") + glob.glob("/usr/local/nvidia/lib64/libcuda.so*"))
    if cu: subprocess.run(["ln", "-sf", cu[0], "/kaggle/working/libcuda/libcuda.so"])
    env.update(PATH=ENV + "/bin:" + env["PATH"], CUDA_HOME=ENV, MPLBACKEND="Agg",
               LIBRARY_PATH="/kaggle/working/libcuda:" + env.get("LIBRARY_PATH", ""),
               LD_LIBRARY_PATH="/kaggle/working/libcuda:" + env.get("LD_LIBRARY_PATH", ""))
    p = subprocess.Popen(cmd, shell=True, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if keep is None or re.search(keep, line): print(line, end="", flush=True)
    return p.wait()

# ---- 1. project (3D PFESA repo) + checkpoints ----
os.makedirs("/kaggle/working/w3d", exist_ok=True)
if not glob.glob("/kaggle/working/w3d/**/train.py", recursive=True):
    zips = glob.glob("/kaggle/input/**/T-Mamba-3D_PFESA_FINAL.zip", recursive=True)
    if zips: subprocess.run(["unzip", "-q", "-o", zips[0], "-d", "/kaggle/working/w3d"], check=True)
    else:
        tr = [p for p in glob.glob("/kaggle/input/**/train.py", recursive=True) if os.path.exists(os.path.join(os.path.dirname(p), "lib/models/TMambaPFESA3D.py"))]
        assert tr, "attach the dataset that has T-Mamba-3D_PFESA_FINAL.zip"
        shutil.copytree(os.path.dirname(tr[0]), "/kaggle/working/w3d/T-Mamba-main", dirs_exist_ok=True)
REPO = os.path.dirname(glob.glob("/kaggle/working/w3d/**/train.py", recursive=True)[0])
P = (glob.glob("/kaggle/input/**/best_TMambaPFESA3D.pth", recursive=True) or [None])[0]
assert P, "best_TMambaPFESA3D.pth not found in your datasets"
B = (glob.glob("/kaggle/input/**/best_TMamba3D.pth", recursive=True) or [None])[0]
print("REPO:", REPO); print("PFESA:", P)

# ---- 2. extra python packages for the 3D code (into the existing env) ----
if sh(PY + " -c 'import monai, SimpleITK, skimage, nibabel, batchgenerators, ml_collections, torchtoolbox'") != 0:
    print("installing 3D packages ...")
    sh(PY + " -m pip install -q batchgenerators==0.25 monai==1.3.0 nibabel==5.2.0 SimpleITK==2.3.1 scikit-image ml_collections==0.1.1 "
       "torchtoolbox==0.1.8.2 ptflops tifffile==2023.7.10 tqdm timm==0.9.12 scipy==1.11.4 pandas gdown 'numpy==1.24.1' 2>&1 | tail -3")
    sh(PY + " -m pip install -q Surface_Distance_Based_Measures==0.1 2>&1 | tail -2")
    sh(PY + " -m pip uninstall -y -q transformers tokenizers; " + PY + " -m pip install -q transformers==4.30.2 tokenizers==0.13.3 2>&1 | tail -2")

# ---- 3. baseline checkpoint (Drive) if you did not upload it; dataset (Drive) ----
if not B:
    os.makedirs(REPO + "/pretrain", exist_ok=True); B = REPO + "/pretrain/best_TMamba3D.pth"
    if not os.path.exists(B): sh(PY + " -m gdown 1WQeyZLiUeiNm8onicyeB5-TipOPZuaHu -O " + B)
print("baseline:", B)
EX = "/kaggle/tmp/ex"
def find_data():
    c = [d for d in glob.glob(EX + "/**/valid", recursive=True) if os.path.isdir(d + "/images") and os.path.isdir(d + "/labels")]
    return os.path.dirname(c[0]) if c else None
DATA = find_data()
if DATA is None:
    os.makedirs(EX, exist_ok=True); arc = "/kaggle/tmp/dataset_archive.bin"
    sh(PY + " -m gdown 1ivIhA222AoFxzTYWVAC3yUFSsg5Waqqu -O " + arc)
    if sh("tar -xf " + arc + " -C " + EX) != 0: sh("unzip -q -o " + arc + " -d " + EX)
    if os.path.exists(arc): os.remove(arc)
    DATA = find_data()
vols = sorted(glob.glob(DATA + "/valid/images/*.nii.gz"))
print("DATA:", DATA, "|", len(vols), "validation volumes")

# ---- 4. helper scripts + inference ----

os.makedirs(REPO + "/tools", exist_ok=True)
open(REPO + "/tools/infer_visualize.py", "w").write(r'''
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
''')
open("/kaggle/working/panel.py", "w").write(r'''
import sys, glob, os, numpy as np
sys.path.insert(0, sys.argv[1])
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lib import utils
data, out = sys.argv[2], sys.argv[3]
force = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "auto" else None
for pb in sorted(glob.glob(out + "/*_pred_base.npy")):
    name = os.path.basename(pb).replace("_pred_base.npy", "")
    raw, _ = utils.load_image_or_label(os.path.join(data, "valid", "images", name + ".nii.gz"), [0.5, 0.5, 0.5], type="image")
    gt, _ = utils.load_image_or_label(os.path.join(data, "valid", "labels", name + ".nii.gz"), [0.5, 0.5, 0.5], type="label")
    gt = gt > 0
    k = force if force is not None else int(np.argmax(gt.sum((0, 1))))
    for tag, title in (("base", "T-Mamba 3D"), ("pfesa", "T-Mamba 3D + PFESA")):
        pred = np.load(os.path.join(out, name + "_pred_" + tag + ".npy")) > 0
        dsc = 2.0 * (pred & gt).sum() / max(pred.sum() + gt.sum(), 1)
        ct = raw[:, :, k].T
        lo, hi = np.percentile(ct, 1), np.percentile(ct, 99.5)
        g = np.clip((ct - lo) / (hi - lo + 1e-8), 0, 1)
        rgb = np.stack([g, g, g], -1); m = pred[:, :, k].T
        rgb[m] = 0.55 * rgb[m] + 0.45 * np.array([1, 0, 0])
        fig, ax = plt.subplots(1, 3, figsize=(18, 6.2))
        ax[0].imshow(rgb); ax[0].set_title("CT + Prediction overlay")
        ax[1].imshow(pred[:, :, k].T, cmap="gray"); ax[1].set_title("Predicted segmentation (slice %d)" % k)
        ax[2].imshow(gt[:, :, k].T, cmap="gray"); ax[2].set_title("Ground truth (binarized)")
        for a in ax: a.axis("off")
        fig.suptitle("%s - %s   (3D DSC %.4f)" % (title, name, dsc), fontsize=14)
        fig.tight_layout(); fig.savefig(os.path.join(out, "%s_%s_panel.png" % (name, tag)), dpi=130); plt.close(fig)
        print("saved", name, tag, "slice", k, "DSC %.4f" % dsc, flush=True)
''')
open("/kaggle/working/render3d.py", "w").write(r'''
import sys, glob, os, numpy as np
sys.path.insert(0, sys.argv[1])
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import SimpleITK as sitk
from lib import utils
data, out = sys.argv[2], sys.argv[3]
def draw(ax, m, title, elev, azim):
    m = np.pad(m[::2, ::2, ::2] > 0, 1)
    if not m.any():
        ax.set_title(title + " (empty)"); ax.set_axis_off(); return
    v, f, _, _ = measure.marching_cubes(m.astype(np.float32), 0.5)
    pc = Poly3DCollection(v[f], linewidths=0); pc.set_facecolor("#f1ead8"); pc.set_edgecolor("none"); ax.add_collection3d(pc)
    s = max(m.shape); ax.set_xlim(0, s); ax.set_ylim(0, s); ax.set_zlim(0, s)
    ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(title, fontsize=10)
for pb in sorted(glob.glob(out + "/*_pred_base.npy")):
    name = os.path.basename(pb).replace("_pred_base.npy", "")
    base = np.load(pb); pf = np.load(pb.replace("_pred_base", "_pred_pfesa"))
    gt, _ = utils.load_image_or_label(os.path.join(data, "valid", "labels", name + ".nii.gz"), [0.5, 0.5, 0.5], type="label")
    for tag, arr in (("base", base), ("pfesa", pf)):
        img = sitk.GetImageFromArray(arr.transpose(2, 1, 0).astype(np.uint8)); img.SetSpacing((0.5, 0.5, 0.5))
        sitk.WriteImage(img, os.path.join(out, name + "_seg_" + tag + ".nii.gz"))
    fig = plt.figure(figsize=(13, 9))
    cols = [("Ground truth", gt), ("T-Mamba3D", base), ("T-Mamba3D + PFESA", pf)]
    for r, (elev, azim, vn) in enumerate([(90, -90, "top view"), (10, -90, "front view")]):
        for c, (t, m) in enumerate(cols):
            ax = fig.add_subplot(2, 3, r * 3 + c + 1, projection="3d"); draw(ax, m, t + " - " + vn, elev, azim)
    fig.suptitle(name); fig.tight_layout(); fig.savefig(os.path.join(out, name + "_3d.png"), dpi=130); plt.close(fig)
    print("rendered", name, flush=True)
''')
print("scripts written")
shutil.rmtree(OUT, ignore_errors=True)
sh(PY + " tools/infer_visualize.py --data %s --base %s --pfesa %s --out %s --n %d" % (DATA, B, P, OUT, N_VOLUMES), keep=r"DSC|saved|Error|error|Traceback")
sh(PY + " /kaggle/working/panel.py %s %s %s %s" % (REPO, DATA, OUT, SLICE), keep=r"saved|Error|error")
sh(PY + " /kaggle/working/render3d.py %s %s %s" % (REPO, DATA, OUT), keep=r"rendered|Error|error")

from IPython.display import Image as _I, display
for f in sorted(glob.glob(OUT + "/*_pfesa_panel.png")) + sorted(glob.glob(OUT + "/*_3d.png")) + sorted(glob.glob(OUT + "/*_compare.png")):
    print(os.path.basename(f)); display(_I(filename=f))
