#!/usr/bin/env python3
"""Augment character images to create a dataset dataset.

Usage:
    python3 src/augment.py --src Fonts20X32 --dst dataset --per_class 100

This script finds images in `--src`, groups them by filename stem (character label),
creates `--dst/<label>/` and writes binarized augmented images until each class has
`--per_class` examples. Augmentations: small random rotation and translation (shift).
"""
import argparse
import os
import random
from pathlib import Path
from PIL import Image


def binarize(img, threshold=128):
    return img.convert('L').point(lambda p: 0 if p < threshold else 255, 'L')


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def generate_for_image(img_path: Path, out_dir: Path, count: int, rot_range: int, shift: int, seed=None):
    if seed is not None:
        random.seed(seed)
    img = Image.open(img_path).convert('L')
    img = binarize(img)
    w, h = img.size
    ensure_dir(out_dir)

    existing = sorted(list(out_dir.glob('*.png')))
    idx = len(existing)

    # include original first
    if idx < count:
        out_path = out_dir / f"{out_dir.name}_{idx:03d}.png"
        img.save(out_path)
        idx += 1

    while idx < count:
        angle = random.uniform(-rot_range, rot_range)
        dx = random.randint(-shift, shift)
        dy = random.randint(-shift, shift)

        # rotate (no expand) with white background
        rotated = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)

        # paste onto white background at offset to avoid wrapping
        canvas = Image.new('L', (w, h), 255)
        canvas.paste(rotated, (dx, dy))

        out_img = binarize(canvas)
        out_path = out_dir / f"{out_dir.name}_{idx:03d}.png"
        out_img.save(out_path)
        idx += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='Source folder with original character images')
    p.add_argument('--dst', required=True, help='Destination training folder (e.g., dataset)')
    p.add_argument('--per_class', type=int, default=20, help='Target examples per class')
    p.add_argument('--rot_range', type=int, default=15, help='Max rotation degrees (+/-)')
    p.add_argument('--shift', type=int, default=3, help='Max pixel shift for x and y (+/-)')
    p.add_argument('--seed', type=int, default=None)
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    ensure_dir(dst)

    # gather image files
    files = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')]
    if not files:
        print(f'No image files found in {src}.')
        return

    # group by filename stem (e.g., "a.png" -> label 'a')
    groups = {}
    for f in files:
        label = f.stem
        groups.setdefault(label, []).append(f)

    for label, paths in groups.items():
        out_dir = dst / label
        ensure_dir(out_dir)
        # if there are no files in this label, skip
        if not paths:
            continue
        # pick images round-robin from available originals to produce variety
        i = 0
        while len(list(out_dir.glob('*.png'))) < args.per_class:
            src_img = paths[i % len(paths)]
            generate_for_image(src_img, out_dir, args.per_class, args.rot_range, args.shift, seed=args.seed)
            i += 1
        print(f'Created {len(list(out_dir.glob("*.png")))} images for label "{label}"')


if __name__ == '__main__':
    main()
