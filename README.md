# 🔬 MediScan AI — Medical Image Diagnostic Platform

> **Hackathon Prototype** — Modular, AI-powered medical scan analysis using pretrained CNNs (ResNet / EfficientNet) and a clean Streamlit UI.

---

## Directory Structure

```
medical_ai_diagnostic/
│
├── app.py                          # Streamlit front-end (entry point)
│
├── modules/
│   ├── __init__.py
│   ├── image_preprocessing.py      # Per-scan-type image transforms
│   ├── model_loader.py             # Model registry + backbone builder
│   └── disease_classifier.py      # Inference engine
│
├── models/                         # Place fine-tuned .pth weights here
│   └── (empty — add your weights)
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

pip install -r requirements.txt
```

> **GPU users:** Install the CUDA-enabled PyTorch first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## How to Use

1. **Select the scan type** in the left sidebar (X-Ray, MRI, CT, Skin Lesion, or Retinal Fundus).
2. **Upload a medical image** — PNG, JPEG, BMP, or TIFF.
3. Press **▶ Run Diagnostic**.
4. View the **primary diagnosis**, **confidence score**, and **full class probability distribution**.

---

## Adding a New Disease Model

The system is designed to be extended with zero changes to the core pipeline.

### Step 1 — Add an entry to `MODEL_REGISTRY` in `modules/model_loader.py`

```python
"Mammography": ModelConfig(
    backbone     = "efficientnet_b0",   # resnet18 | resnet50 | efficientnet_b0
    num_classes  = 4,
    class_names  = [
        "Normal",
        "DCIS (Ductal Carcinoma In Situ)",
        "Invasive Ductal Carcinoma",
        "Benign Mass",
    ],
    weights_path = "models/mammography_efficientnet.pth",  # None = ImageNet only
    description  = "EfficientNet-B0 for mammography cancer screening",
),
```

### Step 2 — Add preprocessing config to `modules/image_preprocessing.py`

```python
"Mammography": {
    "resize": (224, 224),
    "grayscale_to_rgb": True,    # mammograms are grayscale
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225],
    "description": "Grayscale mammogram → 3-channel for CNN input",
},
```

### Step 3 — Drop in your trained weights

Train your model (it must share the same backbone + head architecture) and save:

```python
torch.save(model.state_dict(), "models/mammography_efficientnet.pth")
```

**That's it.** The new scan type will automatically appear in the Streamlit sidebar.

---

## Architecture Overview

```
User uploads image
       │
       ▼
image_preprocessing.py          ← scan-type-aware PIL transforms
  • grayscale → RGB (if needed)
  • resize to 224×224
  • normalize (ImageNet stats)
       │
       ▼
model_loader.py                  ← pluggable model registry
  • loads backbone (ResNet/EfficientNet)
  • replaces final FC layer for N classes
  • loads fine-tuned weights if present
       │
       ▼
disease_classifier.py            ← DiagnosticEngine
  • caches loaded models in memory
  • runs torch.no_grad() inference
  • softmax → sorted (class, prob) pairs
       │
       ▼
app.py (Streamlit)               ← UI layer
  • file upload
  • scan type selector
  • result card + probability bars
```

---

## Supported Scan Types (out of the box)

| Scan Type        | Backbone        | Classes                                                  |
|------------------|-----------------|----------------------------------------------------------|
| X-Ray (Chest)    | ResNet-50       | Normal, Pneumonia, COVID-19, Tuberculosis, Cardiomegaly, Pleural Effusion |
| MRI (Brain)      | ResNet-50       | Normal, Glioma, Meningioma, Pituitary Tumor, Alzheimer's |
| CT (Chest/Abdomen)| EfficientNet-B0| Normal, Lung Cancer, Emphysema, Pulmonary Fibrosis, Pneumothorax |
| Skin Lesion      | EfficientNet-B0 | Benign, Melanoma, BCC, SCC, Actinic Keratosis            |
| Retinal Fundus   | ResNet-18       | Normal, Diabetic Retinopathy, Glaucoma, AMD, Hypertensive Retinopathy |

---

## Training a Real Model (next steps)

1. **Download a public medical dataset** — e.g. NIH Chest X-ray14, ISIC 2019 (skin), BraTS (MRI).
2. Use the backbone from `model_loader._build_backbone()` as your starting point.
3. Fine-tune with standard cross-entropy loss for 10–30 epochs.
4. Save weights: `torch.save(model.state_dict(), "models/<name>.pth")`
5. Set `weights_path` in the registry entry.

---

## ⚠️ Disclaimer

This is a **research prototype** built for a hackathon. The models use ImageNet pretrained weights and have **not been fine-tuned on medical data**. Predictions are **not medically validated** and must not be used for clinical decision-making.
