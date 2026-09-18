#!/usr/bin/env python
"""Port of ``prepare_imgs.py``: select images larger than 512x512 from a glob
pattern and randomly downsample each by a factor in [0.6, 1] (to remove
JPEG compression artifacts before using them as R-D-upper-bound training
data), saving the results as PNGs.

    python scripts/prepare_imgs.py '/tmp/train2017/*.jpg' data/my_coco_train2017/
"""
import argparse
import glob
import os
import random
import sys

from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("glob_pattern")
    p.add_argument("--out_dir")
    p.add_argument("--min_size", type=int, default=512)
    p.add_argument("--min_scale", type=float, default=0.6)
    p.add_argument("--max_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(args.glob_pattern))
    kept = 0
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
        w, h = img.size
        if w < args.min_size or h < args.min_size:
            continue
        scale = random.uniform(args.min_scale, args.max_scale)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        out_path = os.path.join(args.out_dir, os.path.splitext(os.path.basename(path))[0] + ".png")
        img.save(out_path)
        kept += 1
    print(f"Kept {kept}/{len(paths)} images (>= {args.min_size}x{args.min_size}), saved to {args.out_dir}")


if __name__ == "__main__":
    main()
