"""
gradcam.py
----------
Lightweight GradCAM for ResNet (18/50) and EfficientNet-B0 backbones.

Usage:
    from gradcam import generate_gradcam
    heatmap_arr, overlay_pil = generate_gradcam(
        model, image_tensor, target_class_idx, backbone="resnet50", original_image=pil_img
    )
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Layer selector
# ─────────────────────────────────────────────────────────────────────────────
def _get_target_layer(model: nn.Module, backbone: str) -> nn.Module:
    """Return the final convolutional block used as the GradCAM target."""
    b = backbone.lower()
    if "resnet" in b:
        return model.layer4[-1]          # final BasicBlock / Bottleneck
    elif "efficientnet" in b:
        return model.features[-1]        # final MBConv block
    else:
        raise ValueError(f"GradCAM: unsupported backbone '{backbone}'")


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM
# ─────────────────────────────────────────────────────────────────────────────
def generate_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,   # (1, 3, H, W), on model device
    target_class: int,
    backbone: str,
    original_image: Image.Image,
) -> tuple[np.ndarray, Image.Image]:
    """
    Generate a GradCAM heatmap and a blended overlay image.

    Args:
        model:          PyTorch model in eval mode
        image_tensor:   Preprocessed input tensor (1, 3, 224, 224)
        target_class:   Class index to explain
        backbone:       One of 'resnet50', 'resnet18', 'efficientnet_b0'
        original_image: Original PIL Image for the overlay

    Returns:
        heatmap_arr  — float32 numpy array [0, 1] of shape (224, 224)
        overlay_pil  — PIL Image with heatmap blended over original
    """
    activations: list[torch.Tensor] = []
    gradients:   list[torch.Tensor] = []

    target_layer = _get_target_layer(model, backbone)

    def _fwd_hook(module, inp, out):
        activations.append(out)

    def _bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fwd_h = target_layer.register_forward_hook(_fwd_hook)
    bwd_h = target_layer.register_full_backward_hook(_bwd_hook)

    try:
        model.zero_grad()
        output = model(image_tensor)              # (1, num_classes)
        score  = output[0, target_class]
        score.backward()

        grad = gradients[0]                       # (1, C, h, w)
        act  = activations[0]                     # (1, C, h, w)

        # Channel weights via global average pooling of gradients
        weights = grad.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()         # (224, 224)

        # Normalize to [0, 1]
        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo + 1e-8)

        overlay = _blend_heatmap(cam, original_image)
        return cam, overlay

    finally:
        fwd_h.remove()
        bwd_h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Overlay helper
# ─────────────────────────────────────────────────────────────────────────────
def _blend_heatmap(
    cam: np.ndarray,
    original: Image.Image,
    alpha: float = 0.45,
) -> Image.Image:
    """Blend a [0,1] cam array onto the original PIL image using a jet colormap."""
    try:
        import cv2
        orig_rgb  = np.array(original.convert("RGB").resize((224, 224)), dtype=np.uint8)
        heat_u8   = (cam * 255).astype(np.uint8)
        heat_bgr  = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_rgb  = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        blended   = (alpha * heat_rgb + (1 - alpha) * orig_rgb).astype(np.uint8)
        return Image.fromarray(blended)
    except ImportError:
        # Fallback: pure-numpy jet approximation (no cv2)
        return _numpy_jet_overlay(cam, original, alpha)


def _numpy_jet_overlay(cam: np.ndarray, original: Image.Image, alpha: float) -> Image.Image:
    """Pure-numpy jet colourmap fallback when OpenCV is unavailable."""
    r = np.clip(1.5 - np.abs(cam * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(cam * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(cam * 4 - 1), 0, 1)
    heat_rgb  = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    orig_rgb  = np.array(original.convert("RGB").resize((224, 224)), dtype=np.uint8)
    blended   = (alpha * heat_rgb + (1 - alpha) * orig_rgb).astype(np.uint8)
    return Image.fromarray(blended)
