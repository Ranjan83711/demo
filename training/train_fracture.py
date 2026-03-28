import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report

from dataset import load_dataset
from model import get_model

DATA_PATH = "../data/Bone_Fracture_Binary_Classification"
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "../weights/fracture_resnet18.pth"

train_loader, test_loader, classes = load_dataset(DATA_PATH)

print("Classes:", classes)
print("Using device:", DEVICE)

model = get_model(num_classes=len(classes)).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training
for epoch in range(EPOCHS):

    model.train()
    correct, total, loss_sum = 0, 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, labels in loop:

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(acc=100*correct/total)

    print(f"Epoch {epoch+1} Acc: {100*correct/total:.2f}%")

# Evaluation
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(DEVICE))
        _, pred = outputs.max(1)
        y_true.extend(labels.numpy())
        y_pred.extend(pred.cpu().numpy())

print(classification_report(y_true, y_pred, target_names=classes))

os.makedirs("../weights", exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)

print("Saved to:", SAVE_PATH)