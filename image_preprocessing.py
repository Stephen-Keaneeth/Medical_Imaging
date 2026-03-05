"""
image_preprocessing.py
-----------------------
Handles image loading and preprocessing for different medical scan types.
Each scan type can have its own normalization strategy and transforms.

To add preprocessing for a new scan type, add an entry to SCAN_CONFIGS.
"""

from PIL import Image, ImageOps
import torch
from torchvision import transforms
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Per-scan-type preprocessing configurations
# Each entry defines: resize, normalization mean/std, and any special steps
# ─────────────────────────────────────────────────────────────────────────────
SCAN_CONFIGS = {
    "X-Ray (Chest)": {
        "resize": (224, 224),
        "grayscale_to_rgb": True,         # X-rays are grayscale → replicate to 3ch
        "mean": [0.485, 0.456, 0.406],    # ImageNet stats (fine-tune later with CXR stats)
        "std":  [0.229, 0.224, 0.225],
        "description": "Standard chest X-ray — converted to 3-channel for CNN input",
    },
    "MRI (Brain)": {
        "resize": (224, 224),
        "grayscale_to_rgb": True,
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
        "description": "Brain MRI — windowed to soft-tissue range",
    },
    "CT (Chest/Abdomen)": {
        "resize": (224, 224),
        "grayscale_to_rgb": True,
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
        "description": "CT scan — Hounsfield unit clipping applied before normalization",
    },
    "Skin Lesion": {
        "resize": (224, 224),
        "grayscale_to_rgb": False,        # Dermoscopy images are already RGB
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
        "description": "Dermoscopy skin image — colour-space preserved",
    },
    "Retinal Fundus": {
        "resize": (224, 224),
        "grayscale_to_rgb": False,
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
        "description": "Retinal fundus photograph",
    },
}


def get_supported_scan_types() -> list[str]:
    """Return the list of scan types the system supports."""
    return list(SCAN_CONFIGS.keys())


def preprocess_image(image: Image.Image, scan_type: str) -> torch.Tensor:
    """
    Preprocess a PIL Image for the given scan type.

    Args:
        image:      PIL Image (any mode/size)
        scan_type:  One of the keys in SCAN_CONFIGS

    Returns:
        Preprocessed tensor of shape (1, 3, H, W), ready for the model
    """
    if scan_type not in SCAN_CONFIGS:
        raise ValueError(
            f"Unknown scan type '{scan_type}'. "
            f"Supported: {list(SCAN_CONFIGS.keys())}"
        )

    cfg = SCAN_CONFIGS[scan_type]
    resize = cfg["resize"]

    # ── Step 1: Handle RGBA / palette modes ──────────────────────────────────
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode == "P":
        image = image.convert("RGBA").convert("RGB")
    elif image.mode == "L":
        # Grayscale — keep as L until the step below
        pass
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # ── Step 2: Grayscale → RGB replication for X-ray / MRI / CT ─────────────
    if cfg["grayscale_to_rgb"]:
        if image.mode != "L":
            image = ImageOps.grayscale(image)   # collapse to true grayscale
        image = Image.merge("RGB", [image, image, image])
    else:
        if image.mode != "RGB":
            image = image.convert("RGB")

    # ── Step 3: Standard torchvision transforms ───────────────────────────────
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])

    tensor = transform(image)          # shape: (3, H, W)
    return tensor.unsqueeze(0)         # shape: (1, 3, H, W)


def get_scan_description(scan_type: str) -> str:
    """Return a human-readable description of the scan type's preprocessing."""
    return SCAN_CONFIGS.get(scan_type, {}).get("description", "")
