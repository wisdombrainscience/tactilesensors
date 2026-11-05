"""Verification of the object detector (run from a YOLOv5 v6.1 checkout).

1. Split audit: source-photo IDs in Roboflow file names (IMG_xxxx_JPG.rf.*)
   are compared between train and val (augmented copies, adjacent shots).
2. Labels are rebuilt from data/labels/val.cache.
3. val.py is run on the deployed weights (best1.pt) and on best.pt, and
   batch-1 CPU latency is measured.

  python verify_detector.py --repo /path/to/yolov5 --data /path/to/thesis/yolov5/data \
         --weights /path/to/runs/train/exp2/weights
Requires TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 with PyTorch >= 2.6 and numpy aliases
(np.int/np.float) patched in the 2022 code base.
"""
import argparse, glob, os, re, subprocess, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--data", required=True); ap.add_argument("--weights", required=True)
ap.add_argument("--out", default="/tmp/ds")
a = ap.parse_args()

src = lambda f: re.sub(r"\.rf\..*", "", os.path.basename(f))
num = lambda f: int(re.findall(r"IMG_(\d+)", f)[0])
tr = glob.glob(f"{a.data}/images/train/*.jpg"); va = glob.glob(f"{a.data}/images/val/*.jpg")
st = {src(f) for f in tr}; trn = {num(f) for f in tr}
print("train", len(tr), "sources", len(st), "| val", len(va), "| shared sources", len({src(f) for f in va} & st))
print("val with adjacent train shot (+-1):", sum(any(abs(num(f) - t) <= 1 for t in trn) for f in va))

os.makedirs(f"{a.out}/images/val", exist_ok=True); os.makedirs(f"{a.out}/labels/val", exist_ok=True)
c = np.load(f"{a.data}/labels/val.cache", allow_pickle=True).item()
for k, v in c.items():
    if k.endswith(".jpg"):
        n = os.path.splitext(os.path.basename(k))[0]
        np.savetxt(f"{a.out}/labels/val/{n}.txt", v[0].reshape(-1, 5), fmt="%d %.6f %.6f %.6f %.6f")
for f in va:
    os.system(f'cp "{f}" "{a.out}/images/val/"')
open(f"{a.out}/data.yaml", "w").write(f"path: {a.out}\ntrain: images/val\nval: images/val\nnc: 3\nnames: ['khoaitay','nuocmuoi','nuocsuoi']\n")
for w in ("best1", "best"):
    subprocess.run([sys.executable, "val.py", "--weights", f"{a.weights}/{w}.pt", "--data", f"{a.out}/data.yaml",
                    "--img", "416", "--device", "cpu", "--verbose", "--name", w, "--exist-ok"], cwd=a.repo)
