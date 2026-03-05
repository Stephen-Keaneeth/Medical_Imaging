"""
disease_classifier.py
---------------------
Inference engine that ties preprocessing → model → softmax → ranked results.

Usage example:
    from PIL import Image
    from disease_classifier import DiagnosticEngine

    engine = DiagnosticEngine()
    image  = Image.open("chest_xray.png")
    result = engine.predict(image, "X-Ray (Chest)")
    print(result)
"""

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from PIL import Image

from image_preprocessing import preprocess_image
from model_loader import load_model, get_registered_scan_types, ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PredictionResult:
    scan_type: str
    top_class: str
    top_probability: float
    all_classes: list[str]
    all_probabilities: list[float]   # sorted descending
    model_backbone: str
    using_finetuned_weights: bool
    device_used: str

    def top_n(self, n: int = 3) -> list[tuple[str, float]]:
        """Return the top-n (class, probability) pairs."""
        return list(zip(self.all_classes[:n], self.all_probabilities[:n]))

    def __str__(self) -> str:
        lines = [
            f"Scan type : {self.scan_type}",
            f"Top class : {self.top_class}  ({self.top_probability*100:.1f}%)",
            "─" * 40,
        ]
        for cls, prob in zip(self.all_classes, self.all_probabilities):
            bar = "█" * int(prob * 30)
            lines.append(f"  {cls:<30} {prob*100:5.1f}%  {bar}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic engine
# ─────────────────────────────────────────────────────────────────────────────
class DiagnosticEngine:
    """
    Manages model caching and runs inference for any registered scan type.

    Models are loaded lazily and cached in memory so repeated calls for the
    same scan type don't reload weights from disk.
    """

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model_cache: dict[str, tuple] = {}   # scan_type → (model, config)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_model(self, scan_type: str):
        """Load (and cache) the model for a scan type."""
        if scan_type not in self._model_cache:
            model, cfg = load_model(scan_type, self.device)
            self._model_cache[scan_type] = (model, cfg)
        return self._model_cache[scan_type]

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, image: Image.Image, scan_type: str) -> PredictionResult:
        """
        Run inference on a PIL Image for the specified scan type.

        Args:
            image:     PIL Image (any size / mode)
            scan_type: One of the registered scan types

        Returns:
            PredictionResult with ranked classes and probabilities
        """
        if scan_type not in get_registered_scan_types():
            raise ValueError(
                f"'{scan_type}' is not a registered scan type. "
                f"Supported: {get_registered_scan_types()}"
            )

        # Preprocessing
        tensor = preprocess_image(image, scan_type)      # (1, 3, H, W)
        tensor = tensor.to(self.device)

        # Inference
        model, cfg = self._get_model(scan_type)
        with torch.no_grad():
            logits = model(tensor)                        # (1, num_classes)
            probs  = F.softmax(logits, dim=1).squeeze()  # (num_classes,)

        # Sort descending
        sorted_indices = torch.argsort(probs, descending=True).tolist()
        sorted_classes = [cfg.class_names[i] for i in sorted_indices]
        sorted_probs   = [probs[i].item()    for i in sorted_indices]

        return PredictionResult(
            scan_type              = scan_type,
            top_class              = sorted_classes[0],
            top_probability        = sorted_probs[0],
            all_classes            = sorted_classes,
            all_probabilities      = sorted_probs,
            model_backbone         = cfg.backbone,
            using_finetuned_weights= bool(cfg.weights_path),
            device_used            = str(self.device),
        )

    def supported_scan_types(self) -> list[str]:
        return get_registered_scan_types()

    def warm_up(self, scan_type: str) -> None:
        """Pre-load the model for a scan type to reduce first-inference latency."""
        self._get_model(scan_type)
        print(f"[DiagnosticEngine] Warmed up model for: {scan_type}")
