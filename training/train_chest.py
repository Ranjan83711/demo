import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report

from dataset import load_dataset
from model import get_model

# -------------------------
# Config
# -------------------------
DATA_PATH = "../data/chest_xray"
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "../weights/chest_xray_resnet18.pth"

# -------------------------
# Load Data
# -------------------------
train_loader, test_loader, classes = load_dataset(DATA_PATH)

print("Classes:", classes)
print("Using device:", DEVICE)

# -------------------------
# Model
# -------------------------
model = get_model(num_classes=len(classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Training Loop
# -------------------------
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, labels in loop:

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(
            loss=total_loss/(total+1),
            acc=100.*correct/total
        )

    print(f"\nEpoch {epoch+1} Training Accuracy: {100.*correct/total:.2f}%")

# -------------------------
# Evaluation
# -------------------------
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\nTest Report:\n")
print(classification_report(y_true, y_pred, target_names=classes))

# -------------------------
# Save Model
# -------------------------
os.makedirs("../weights", exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)

print(f"\nModel saved to {SAVE_PATH}")