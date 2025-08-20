import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch import nn

early_stopping_patience = 10
epochs_without_improvement = 0



log_file = open("metrics_log.txt", "w")
log_file.write("Epoch,TrainLoss,TrainAcc,ValLoss,ValAcc,Precision,Recall,F1\n")



# Custom Dataset

class GenderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.samples = []

        for fname in os.listdir(folder_path):
            if fname.endswith('.jpg') or fname.endswith('.png'):
                label_file = os.path.join(folder_path, os.path.splitext(fname)[0] + ".txt")
                if os.path.exists(label_file):
                    with open(label_file, "r") as f:
                        lines = f.readlines()
                        gender_line = [l for l in lines if "gender" in l.lower()]
                        if gender_line:
                            gender = int(gender_line[0].split(":")[1].strip())
                            self.samples.append((fname, gender))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, gender = self.samples[idx]
        image_path = os.path.join(self.folder_path, fname)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, gender


# Transforms

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
])


val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Load Dataset

train_dataset = GenderDataset(r"D:\gender_only_data\train", transform=train_transforms)
val_dataset = GenderDataset(r"D:\gender_only_data\val", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)


# Load Model


from torchvision.models import ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNet50WithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, x):
        return self.base(x)

model = ResNet50WithDropout().to(device)


# Loss, Optimizer, Scheduler

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# Training Loop

best_val_f1 = 0
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(1, 151):
    model.train()
    train_loss = 0
    train_preds, train_targets = [], []

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.cpu().numpy())
        train_targets.extend(labels.cpu().numpy())

    train_acc = accuracy_score(train_targets, train_preds)
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_loss = 0
    val_preds, val_targets = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch:03d} [Val]", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_targets, val_preds)
    val_precision = precision_score(val_targets, val_preds)
    val_recall = recall_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds)

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_acc)

    log_str = (f"Epoch {epoch:03d} | Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_acc:.4f} | "
           f"Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_acc:.4f} | "
           f"Prec: {val_precision:.4f} | Rec: {val_recall:.4f} | F1: {val_f1:.4f}")
    print(log_str)
    log_file.write(f"{epoch},{train_losses[-1]:.4f},{train_acc:.4f},{val_losses[-1]:.4f},"
                f"{val_acc:.4f},{val_precision:.4f},{val_recall:.4f},{val_f1:.4f}\n")


    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_resnet50_gender_model.pth")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch} due to no improvement.")
            break

        
    scheduler.step()

log_file.close()


# Plotting Metrics

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_plot.png")

plt.figure()
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_plot.png")
