"""Live demo: your X-ray image(s) -> segmented mask (T-Mamba2D / UniMiSS-2D).
Put this file in the repo root (next to test_2d.py). Run:
  python demo_infer_2d.py --weight ./runs/<run>/checkpoints/best_TMamba2D.pth --input my_xray.png
  python demo_infer_2d.py --weight <pth> --input ./my_images_folder --out ./demo_out
Outputs per image in --out:  <name>_mask.png (0/255, original size), <name>_overlay.png (red mask on X-ray),
                             <name>_side_by_side.png (input | mask | overlay)"""
import os, sys, glob, argparse, time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

P = {"seed": 1777777, "cuda": True, "benchmark": False, "deterministic": True,
     "resize_shape": (640, 1280),
     "normalize_means": (0.50297405, 0.54711632, 0.71049083),
     "normalize_stds": (0.18653496, 0.17118206, 0.17080363),
     "model_name": "TMamba2D", "dataset_name": "Tooth2D-X-Ray-6k", "dimension": "2d",
     "in_channels": 3, "classes": 2, "high_frequency": 0.9, "low_frequency": 0.1,
     "scaling_version": "SMALL", "multi_gpu": False, "local_rank": 0, "pretrain": None, "save_dir": None}
EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True, help="trained .pth")
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--out", default="./demo_out")
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    from lib import utils, models
    from lib.utils.checkpoint import load_pretrained_flexible

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        print("[warn] no GPU found - Mamba CUDA kernels may fail on CPU. Use the GPU machine for the live demo.")
    P["device"] = dev
    utils.reproducibility(P["seed"], P["deterministic"], P["benchmark"])  # same RNG order as test_2d.py
    model = models.get_model(P)
    load_pretrained_flexible(model, a.weight)
    model = model.to(dev).eval()

    tf = transforms.Compose([transforms.Resize(P["resize_shape"]), transforms.ToTensor(),
                             transforms.Normalize(mean=P["normalize_means"], std=P["normalize_stds"])])
    if os.path.isdir(a.input):
        files = sorted(f for f in glob.glob(os.path.join(a.input, "*")) if f.lower().endswith(EXT))
    else:
        files = [a.input]
    os.makedirs(a.out, exist_ok=True)
    print("images:", len(files))

    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        img = Image.open(f).convert("RGB")
        w, h = img.size
        x = tf(img).unsqueeze(0).to(dev)
        t0 = time.time()
        with torch.no_grad():
            pred = torch.argmax(model(x), dim=1).squeeze(0).to(torch.uint8).cpu().numpy()
        dt = time.time() - t0
        mask = Image.fromarray((pred * 255).astype(np.uint8), mode="L").resize((w, h), Image.NEAREST)
        m = np.array(mask) > 0
        ov = np.array(img).astype(np.float32)
        ov[m] = 0.55 * ov[m] + 0.45 * np.array([255, 0, 0], np.float32)
        ov = Image.fromarray(ov.astype(np.uint8))
        mask.save(os.path.join(a.out, name + "_mask.png"))
        ov.save(os.path.join(a.out, name + "_overlay.png"))
        sbs = Image.new("RGB", (w * 3, h))
        sbs.paste(img, (0, 0)); sbs.paste(mask.convert("RGB"), (w, 0)); sbs.paste(ov, (2 * w, 0))
        sbs.save(os.path.join(a.out, name + "_side_by_side.png"))
        print("{}  ({:.2f}s)  teeth pixels: {:.1f}%".format(name, dt, 100 * m.mean()))
    print("saved to", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
