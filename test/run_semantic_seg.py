#!/usr/bin/env python3
"""
Semantic segmentation on a single RGB image.

Default backend: torchvision DeepLabV3-ResNet50 (Pascal VOC 21 classes).
Optional: HuggingFace SegFormer-B2 Cityscapes 19 classes (--backend segformer)
          requires network access to huggingface.co

Outputs under test/output/:
  <stem>_mask.png, <stem>_overlay.png, <stem>_labels.txt

Usage:
  python test/run_semantic_seg.py
  python test/run_semantic_seg.py --image path/to/frame.png
  python test/run_semantic_seg.py --backend segformer   # Cityscapes, needs HF
"""

from __future__ import annotations

import argparse
import importlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = (
    ROOT
    / "samples"
    / "paper_eval_l2_sem_rich_full_preview"
    / "uav"
    / "front_rgb"
    / "000000.png"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SEGFORMER_ID = "nvidia/segformer-b2-finetuned-cityscapes-512-1024"


def _labels_module(backend: str):
    if backend == "segformer":
        return importlib.import_module("cityscapes_labels")
    return importlib.import_module("voc_labels")


def load_torchvision_deeplab(device: torch.device):
    from torchvision.models.segmentation import (
        DeepLabV3_ResNet50_Weights,
        deeplabv3_resnet50,
    )

    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    model.eval().to(device)
    preprocess = weights.transforms()
    return model, preprocess, _labels_module("voc")


def load_segformer(device: torch.device):
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    processor = AutoImageProcessor.from_pretrained(SEGFORMER_ID)
    model = AutoModelForSemanticSegmentation.from_pretrained(SEGFORMER_ID)
    model.eval().to(device)
    return processor, model, _labels_module("segformer")


@torch.inference_mode()
def predict_torchvision(image: Image.Image, model, preprocess, device) -> np.ndarray:
    batch = preprocess(image).unsqueeze(0).to(device)
    out = model(batch)["out"]
    pred = out.argmax(dim=1).unsqueeze(0).float()
    pred = torch.nn.functional.interpolate(
        pred,
        size=image.size[::-1],  # (H, W)
        mode="nearest",
    )
    return pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.int32)


@torch.inference_mode()
def predict_segformer(image: Image.Image, processor, model, device) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)


def save_results(
    image_path: Path,
    image: Image.Image,
    mask: np.ndarray,
    labels_mod,
    out_dir: Path,
    backend: str,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    rgb = np.array(image.convert("RGB"))
    color_mask = labels_mod.mask_to_color(mask)
    overlay = (0.5 * rgb + 0.5 * color_mask).astype(np.uint8)

    paths = {
        "mask": out_dir / f"{stem}_mask.png",
        "overlay": out_dir / f"{stem}_overlay.png",
        "labels": out_dir / f"{stem}_labels.txt",
    }
    Image.fromarray(color_mask).save(paths["mask"])
    Image.fromarray(overlay).save(paths["overlay"])

    counts = Counter(mask.flatten().tolist())
    total = mask.size
    id2label = labels_mod.ID2LABEL
    lines = [
        f"image: {image_path}",
        f"backend: {backend}",
        f"size: {mask.shape[1]}x{mask.shape[0]}",
        "",
    ]
    for tid in sorted(counts.keys()):
        name = id2label.get(tid, f"id_{tid}")
        n = counts[tid]
        pct = 100.0 * n / total
        if pct >= 0.05:
            lines.append(f"{name:16s}  {pct:5.1f}%  ({n} px)")
    paths["labels"].write_text("\n".join(lines), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic segmentation demo")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--backend",
        choices=("deeplab", "segformer"),
        default="deeplab",
        help="deeplab=VOC21 (offline); segformer=Cityscapes19 (needs HF)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    image_path = args.image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    device = torch.device(args.device)
    image = Image.open(image_path).convert("RGB")

    if args.backend == "segformer":
        print(f"Loading SegFormer {SEGFORMER_ID} on {device} ...")
        processor, model, labels_mod = load_segformer(device)
        print(f"Inference: {image_path.name} ({image.size[0]}x{image.size[1]})")
        mask = predict_segformer(image, processor, model, device)
    else:
        print(f"Loading DeepLabV3-ResNet50 (VOC) on {device} ...")
        model, preprocess, labels_mod = load_torchvision_deeplab(device)
        print(f"Inference: {image_path.name} ({image.size[0]}x{image.size[1]})")
        mask = predict_torchvision(image, model, preprocess, device)

    paths = save_results(
        image_path, image, mask, labels_mod, args.out_dir.resolve(), args.backend,
    )
    print("Saved:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    print("\n" + paths["labels"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
