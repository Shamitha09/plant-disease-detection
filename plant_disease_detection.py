import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, classification_report

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_DIR = "plantvillage/plantvillage dataset/color"  
BATCH_SIZE = 32
EPOCHS = 3            # bump to 5-8 if you have time; diminishing returns after that
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
IMG_SIZE = 224         # ResNet18's expected input size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cpu":
    print("WARNING: No GPU detected. Go to Runtime > Change runtime type > GPU, then restart.")

# ---------------------------------------------------------------------
# 1. DATA LOADING & AUGMENTATION
# ---------------------------------------------------------------------
# ImageNet mean/std — required since we're using ImageNet-pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Found {len(full_dataset)} images across {num_classes} classes")

val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
# validation set should use the non-augmented transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ---------------------------------------------------------------------
# 2. MODEL — ResNet18 pretrained on ImageNet, fine-tuned for our classes
# ---------------------------------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze early layers (keep their learned general features), only train the
# last block + final classifier — faster and needs less data than full fine-tuning
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, num_classes)  # replace final layer for our #classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# ---------------------------------------------------------------------
# 3. TRAINING LOOP
# ---------------------------------------------------------------------
best_val_acc = 0.0
best_model_weights = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    start = time.time()

    # --- train ---
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = running_correct / total

    # --- validate ---
    model.eval()
    val_correct, val_total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / val_total
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    scheduler.step()

    elapsed = time.time() - start
    print(f"Epoch {epoch+1}/{EPOCHS} ({elapsed:.0f}s) | "
          f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | "
          f"Val acc: {val_acc:.4f}, Val F1 (macro): {val_f1:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_weights = copy.deepcopy(model.state_dict())

model.load_state_dict(best_model_weights)

# ---------------------------------------------------------------------
# 4. FINAL REPORT
# ---------------------------------------------------------------------
print(f"\n=== Best Validation Accuracy: {best_val_acc:.4f} ===")
print("\n=== Classification Report (last epoch's predictions) ===")
print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

# ---------------------------------------------------------------------
# 5. SAVE MODEL
# ---------------------------------------------------------------------
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
}, "plant_disease_model.pth")
print("\nModel saved to plant_disease_model.pth")
