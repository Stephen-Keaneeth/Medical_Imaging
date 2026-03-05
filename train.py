import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import random

from model_loader import _build_backbone


def main():

    print("===== MediScan AI Training Started =====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # dataset paths
    train_dir = "data/chest_xray/train"
    val_dir = "data/chest_xray/val"

    if not os.path.exists(train_dir):
        print("ERROR: train dataset not found at", train_dir)
        return

    if not os.path.exists(val_dir):
        print("ERROR: val dataset not found at", val_dir)
        return

    # image transforms — must match image_preprocessing.py normalization exactly
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading datasets...")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    print("Classes:", train_dataset.classes)
    print("Training images (full):", len(train_dataset))
    print("Validation images (full):", len(val_dataset))

    # ── Speed optimisation for CPU demo ───────────────────────────────────────
    # Cap at 500 training images and 100 val images so training finishes in
    # ~5-8 minutes on CPU. Remove these lines for full production training.
    MAX_TRAIN = 500
    MAX_VAL   = 100
    if len(train_dataset) > MAX_TRAIN:
        indices = random.sample(range(len(train_dataset)), MAX_TRAIN)
        train_dataset = Subset(train_dataset, indices)
        print(f"Using {MAX_TRAIN}-image subset for fast CPU training.")
    if len(val_dataset) > MAX_VAL:
        indices = random.sample(range(len(val_dataset)), MAX_VAL)
        val_dataset = Subset(val_dataset, indices)
    # ──────────────────────────────────────────────────────────────────────────

    # Warn if class count doesn't match the registry entry (model_loader.py expects 6)
    REGISTRY_NUM_CLASSES = 6
    orig_classes = train_dataset.dataset.classes if isinstance(train_dataset, Subset) else train_dataset.classes
    if len(orig_classes) != REGISTRY_NUM_CLASSES:
        print(
            f"WARNING: Dataset has {len(orig_classes)} class folder(s), but "
            f"model_loader.py registers 'X-Ray (Chest)' with num_classes={REGISTRY_NUM_CLASSES}.\n"
            f"Update MODEL_REGISTRY in model_loader.py to match your dataset classes, "
            f"or reorganise your data folders."
        )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    print("Building model...")

    # resnet18 trains ~3x faster than resnet50 and is plenty for a hackathon demo.
    # Switch to "resnet50" for higher accuracy with a full dataset.
    BACKBONE = "resnet18"
    model = _build_backbone(BACKBONE, num_classes=len(orig_classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 2   # 2 epochs is enough for a demo; increase to 10+ for production

    print("Training started...\n")

    for epoch in range(epochs):

        model.train()

        total = 0
        correct = 0
        running_loss = 0
        num_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print progress every 10 batches so it doesn't look frozen
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"  Epoch {epoch+1}/{epochs}  Batch {batch_idx+1}/{num_batches}  "
                      f"Loss: {loss.item():.4f}  Acc: {correct/total:.4f}", flush=True)

        train_acc = correct / total

        # validation
        model.eval()

        val_total = 0
        val_correct = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {running_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print("")

    print("Training complete!")

    os.makedirs("models", exist_ok=True)

    # Path matches the registry entry in model_loader.py
    model_path = "models/cxr_resnet50.pth"
    torch.save(model.state_dict(), model_path)

    print("Model saved to:", model_path)


if __name__ == "__main__":
    main()