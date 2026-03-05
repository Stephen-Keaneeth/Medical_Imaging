"""
model_loader.py
---------------
Central model registry.  Each scan type maps to:
  • a backbone architecture  (ResNet50 / EfficientNet-B0 / …)
  • a list of disease classes
  • an optional path to fine-tuned weights (.pth file)

────────────────────────────────────────────────────────────────────────────
HOW TO ADD A NEW DISEASE MODEL
────────────────────────────────────────────────────────────────────────────
1. Add an entry to MODEL_REGISTRY below with:
      "Your Scan Type": ModelConfig(
          backbone     = "resnet50",          # or "efficientnet_b0"
          num_classes  = <N>,
          class_names  = ["Class A", "Class B", ...],
          weights_path = "models/your_weights.pth",  # None → ImageNet pretrain
      )
2. Train / obtain a .pth file and place it in the models/ directory.
3. That's it — the rest of the pipeline picks it up automatically.
────────────────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models


# ─────────────────────────────────────────────────────────────────────────────
# Model configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    backbone: str                        # "resnet50" | "efficientnet_b0" | "resnet18"
    num_classes: int
    class_names: list[str]
    weights_path: Optional[str] = None  # None → use ImageNet pretrained weights only
    description: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Disease class definitions per scan type
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, ModelConfig] = {

    "X-Ray (Chest)": ModelConfig(
        backbone     = "resnet50",
        num_classes  = 6,
        class_names  = [
            "Normal",
            "Pneumonia",
            "COVID-19",
            "Tuberculosis",
            "Cardiomegaly",
            "Pleural Effusion",
        ],
        weights_path = None,   # Replace with "models/cxr_resnet50.pth" after training
        description  = "ResNet-50 trained on CheXpert / NIH Chest X-ray datasets",
    ),

    "MRI (Brain)": ModelConfig(
        backbone     = "resnet50",
        num_classes  = 5,
        class_names  = [
            "Normal",
            "Glioma",
            "Meningioma",
            "Pituitary Tumor",
            "Alzheimer's Disease",
        ],
        weights_path = None,
        description  = "ResNet-50 for brain MRI tumour classification",
    ),

    "CT (Chest/Abdomen)": ModelConfig(
        backbone     = "efficientnet_b0",
        num_classes  = 5,
        class_names  = [
            "Normal",
            "Lung Cancer (Nodule)",
            "Emphysema",
            "Pulmonary Fibrosis",
            "Pneumothorax",
        ],
        weights_path = None,
        description  = "EfficientNet-B0 for chest/abdominal CT analysis",
    ),

    "Skin Lesion": ModelConfig(
        backbone     = "efficientnet_b0",
        num_classes  = 5,
        class_names  = [
            "Benign / Normal",
            "Melanoma",
            "Basal Cell Carcinoma",
            "Squamous Cell Carcinoma",
            "Actinic Keratosis",
        ],
        weights_path = None,
        description  = "EfficientNet-B0 trained on ISIC dermoscopy dataset",
    ),

    "Retinal Fundus": ModelConfig(
        backbone     = "resnet18",
        num_classes  = 5,
        class_names  = [
            "Normal",
            "Diabetic Retinopathy",
            "Glaucoma",
            "Age-related Macular Degeneration",
            "Hypertensive Retinopathy",
        ],
        weights_path = None,
        description  = "ResNet-18 for retinal fundus disease classification",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal: backbone builders
# ─────────────────────────────────────────────────────────────────────────────
def _build_backbone(name: str, num_classes: int) -> nn.Module:
    """
    Build a backbone CNN with the final layer replaced for num_classes outputs.
    Uses ImageNet pretrained weights as a base (transfer learning).
    """
    name = name.lower()

    if name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    elif name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    elif name == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = net.classifier[1].in_features
        net.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Supported: resnet18, resnet50, efficientnet_b0"
        )

    return net


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def load_model(scan_type: str, device: torch.device) -> tuple[nn.Module, ModelConfig]:
    """
    Load and return (model, config) for the given scan type.

    • If config.weights_path points to an existing .pth file, those fine-tuned
      weights are loaded.
    • Otherwise the backbone's ImageNet pretrained weights are used (the custom
      head will have random weights — predictions are indicative of the
      architecture, not medically validated).

    Args:
        scan_type: Key from MODEL_REGISTRY
        device:    torch.device to move the model to

    Returns:
        (model in eval mode, ModelConfig)
    """
    if scan_type not in MODEL_REGISTRY:
        raise KeyError(
            f"No model registered for scan type '{scan_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cfg = MODEL_REGISTRY[scan_type]
    model = _build_backbone(cfg.backbone, cfg.num_classes)

    # Load fine-tuned weights if a path is provided and the file exists
    if cfg.weights_path:
        import os
        if os.path.isfile(cfg.weights_path):
            state = torch.load(cfg.weights_path, map_location=device)
            model.load_state_dict(state)
            print(f"[model_loader] Loaded fine-tuned weights: {cfg.weights_path}")
        else:
            print(
                f"[model_loader] ⚠ weights_path '{cfg.weights_path}' not found. "
                f"Using ImageNet pretrained weights."
            )

    model.to(device)
    model.eval()
    return model, cfg


def get_registered_scan_types() -> list[str]:
    """Return all scan types that have a registered model."""
    return list(MODEL_REGISTRY.keys())


def get_model_config(scan_type: str) -> ModelConfig:
    """Return the ModelConfig for a scan type without building the model."""
    if scan_type not in MODEL_REGISTRY:
        raise KeyError(f"Scan type '{scan_type}' not in registry.")
    return MODEL_REGISTRY[scan_type]
