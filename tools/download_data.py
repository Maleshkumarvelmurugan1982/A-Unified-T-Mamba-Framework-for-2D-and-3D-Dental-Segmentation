#!/usr/bin/env python3
"""Download and arrange the datasets (and optional baseline checkpoint) inside this repo.

  python tools/download_data.py --what 2d        # Tooth2D-X-Ray-6k  -> datasets/Tooth2D-X-Ray-6k/{train,test}/{images,annotations}
  python tools/download_data.py --what 3d        # CBCT dataset      -> datasets/NC-release-data-checked/{train,valid}/{images,labels}
  python tools/download_data.py --what ckpt3d    # 3D baseline .pth  -> pretrain/best_TMamba3D.pth
  python tools/download_data.py --what all

Needs:  pip install gdown        (Internet; Google Drive links below). Safe to re-run: finished parts are skipped.
If a Drive quota error appears, download the file manually in the browser, put it in  datasets/_downloads/  with the
same name shown below and run the command again."""
import argparse, os, shutil, sys, tarfile, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "datasets")
DL = os.path.join(DS, "_downloads")
LINKS = {
    "2d_part1": "1YTa8Z5iv_pO0G-0KeU-yTritkodg7TMK",   # Tooth2D-X-Ray-6k, part 1
    "2d_part2": "1CUq-MuMIr7eHKBVZ4VTh-m72pcLMhgp2",   # Tooth2D-X-Ray-6k, part 2
    "3d_data":  "1ivIhA222AoFxzTYWVAC3yUFSsg5Waqqu",   # CBCT dataset (NC-release-data-checked)
    "3d_ckpt":  "1WQeyZLiUeiNm8onicyeB5-TipOPZuaHu",   # baseline T-Mamba3D checkpoint
}


def fetch(name, fid):
    os.makedirs(DL, exist_ok=True)
    out = os.path.join(DL, name)
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        print("have", out); return out
    try:
        import gdown
    except ImportError:
        sys.exit("gdown missing: pip install gdown")
    print("downloading", name, "...")
    gdown.download(id=fid, output=out, quiet=False)
    if not (os.path.isfile(out) and os.path.getsize(out) > 0):
        sys.exit("download failed for %s - download it manually from https://drive.google.com/file/d/%s/view and save it as %s" % (name, fid, out))
    return out


def extract(path, dest):
    """True if `path` was a complete archive and is extracted; False for anything else (e.g. half of a split archive)"""
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z: z.extractall(dest)
            return True
        if tarfile.is_tarfile(path):
            with tarfile.open(path) as t: t.extractall(dest)
            return True
    except (zipfile.BadZipFile, tarfile.TarError, EOFError, OSError) as e:
        print("  (%s is not a complete archive: %s)" % (os.path.basename(path), str(e)[:60]))
    return False


def find_root(base, need):
    """first folder under base that contains every sub-path in `need`"""
    for d, _, _ in os.walk(base):
        if all(os.path.isdir(os.path.join(d, n)) for n in need):
            return d
    return None


def place(src, dst):
    if os.path.isdir(dst): shutil.rmtree(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)


def count(d):
    return len([f for f in os.listdir(d) if not f.startswith(".")]) if os.path.isdir(d) else 0


def get_2d():
    dst = os.path.join(DS, "Tooth2D-X-Ray-6k"); need = ["train/images", "train/annotations", "test/images", "test/annotations"]
    if find_root(dst, need) == dst:
        print("2D dataset already in place"); return
    parts = [fetch("2d_part1", LINKS["2d_part1"]), fetch("2d_part2", LINKS["2d_part2"])]
    ex = os.path.join(DL, "ex2d"); shutil.rmtree(ex, ignore_errors=True); os.makedirs(ex)
    for p in parts:
        if not extract(p, ex): print("part is not an archive by itself:", os.path.basename(p))
    root = find_root(ex, need)
    if root is None:   # the two files may be halves of one archive
        comb = os.path.join(DL, "2d_combined")
        with open(comb, "wb") as o:
            for p in parts:
                with open(p, "rb") as f: shutil.copyfileobj(f, o)
        extract(comb, ex); root = find_root(ex, need); os.remove(comb)
    if root is None: sys.exit("could not find train/test folders after extraction (look inside %s)" % ex)
    place(root, dst); shutil.rmtree(ex, ignore_errors=True)
    for sp in ("train", "test"):
        print("2D %s: %d images, %d annotations" % (sp, count(dst + "/%s/images" % sp), count(dst + "/%s/annotations" % sp)))
    print("2D DATASET OK ->", dst)


def get_3d():
    dst = os.path.join(DS, "NC-release-data-checked"); need = ["train/images", "train/labels", "valid/images", "valid/labels"]
    if find_root(dst, need) == dst:
        print("3D dataset already in place"); return
    arc = fetch("3d_data", LINKS["3d_data"])
    ex = os.path.join(DL, "ex3d"); shutil.rmtree(ex, ignore_errors=True); os.makedirs(ex)
    if not extract(arc, ex): sys.exit("3D download is not a zip/tar archive")
    root = find_root(ex, need)
    if root is None: sys.exit("could not find train/valid folders after extraction (look inside %s)" % ex)
    place(root, dst); shutil.rmtree(ex, ignore_errors=True)
    shutil.rmtree(os.path.join(dst, "sub_volumes"), ignore_errors=True)   # stale cache from other machines
    for sp in ("train", "valid"):
        print("3D %s: %d volumes, %d labels" % (sp, count(dst + "/%s/images" % sp), count(dst + "/%s/labels" % sp)))
    print("3D DATASET OK ->", dst)


def get_ckpt3d():
    out = os.path.join(ROOT, "pretrain", "best_TMamba3D.pth")
    if os.path.isfile(out): print("have", out); return
    tmp = fetch("3d_ckpt", LINKS["3d_ckpt"]); os.makedirs(os.path.dirname(out), exist_ok=True); shutil.copy(tmp, out)
    print("checkpoint ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["2d", "3d", "ckpt3d", "all"], default="all")
    w = ap.parse_args().what
    if w in ("2d", "all"): get_2d()
    if w in ("3d", "all"): get_3d()
    if w in ("ckpt3d", "all"): get_ckpt3d()
    print("done. You can delete", DL, "to free disk space.")
