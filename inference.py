#!/usr/bin/env python3
"""Run georeferenced multimodal inference for one PlanetScope patch."""

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch

from dataset.lands2 import PSLandslideSentinel2Dataset
from dataset.landslides import PSLandslideDataset
from dataset.multidata import MultiModalLandslideDataset
from models.swinunet import ChangeDetectionSwinUNet


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--patch-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float)
    return parser.parse_args()


def load_sample(dataset_root, event, patch_id, patch_size):
    planet = PSLandslideDataset(
        dataset_root,
        [event],
        patch_size=patch_size,
        apply_transform=False,
        patch_ids=[patch_id],
    )
    sentinel = PSLandslideSentinel2Dataset(
        dataset_root,
        [event],
        apply_transform=False,
        patch_ids=[patch_id],
    )
    dataset = MultiModalLandslideDataset(planet, sentinel, apply_transform=False)
    for index, item in enumerate(dataset.aligned_indices):
        if item["patch_id"] == patch_id:
            return dataset[index]
    raise KeyError(f"Patch {event}/{patch_id} is not available in the multimodal dataset")


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    required = {"model_state_dict", "config", "best_threshold"}
    if not required.issubset(checkpoint):
        raise ValueError("Checkpoint must be a complete best_model.pth produced by train.py")
    config = checkpoint["config"]
    model = ChangeDetectionSwinUNet(
        img_size=config["patch_size"], model_size=config["model_size"], pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict(model, sample):
    with torch.no_grad():
        logits = model(
            sample["s2_pre"].unsqueeze(0), sample["s2_post"].unsqueeze(0),
            sample["planet_pre"].unsqueeze(0), sample["planet_post"].unsqueeze(0),
            sample["aux"].unsqueeze(0),
            valid_t1=sample["s2_valid_pre"].unsqueeze(0),
            valid_t2=sample["s2_valid_post"].unsqueeze(0),
        )
    return torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)


def write_outputs(reference, output_dir, probability, threshold, checkpoint_path):
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    binary = (probability >= threshold).astype(np.uint8)
    with rasterio.open(reference) as source:
        profile = source.profile.copy()
        expected_shape = (source.height, source.width)
    if probability.shape != expected_shape:
        raise ValueError(
            f"Prediction shape {probability.shape} does not match reference {expected_shape}"
        )
    profile.update(count=1, dtype="float32", nodata=None)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data, dtype in (("probability.tif", probability, "float32"),
                              ("mask.tif", binary, "uint8")):
        output_profile = profile.copy()
        output_profile.update(dtype=dtype)
        with rasterio.open(output_dir / name, "w", **output_profile) as output:
            output.write(data, 1)
            output.update_tags(threshold=str(threshold), checkpoint=str(checkpoint_path))


def main():
    args = parse_args()
    model, checkpoint = load_model(args.checkpoint)
    config = checkpoint["config"]
    threshold = args.threshold if args.threshold is not None else checkpoint["best_threshold"]
    sample = load_sample(args.dataset_root, args.event, args.patch_id, config["patch_size"])
    probability = predict(model, sample)
    reference = args.dataset_root / args.event / args.patch_id / "pre.tif"
    write_outputs(reference, args.output_dir, probability, threshold, args.checkpoint)


if __name__ == "__main__":
    main()
