import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn

from model import CNN4Conv

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_DIR = "./data/simpsons/archive/characters_train"
pic_size = 64  # resize images to 64x64

# gather all class names
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
class_to_idx = {name: i for i, name in enumerate(class_names)}
idx_to_class = {i: name for name, i in class_to_idx.items()}


def get_data_opencv_with_map(directory: str, BGR: bool = False):
    """
    Load all images folder by folder using OpenCV and assign numeric labels.
    Returns:
        images: np.ndarray of shape (N, H, W, 3)
        labels: np.ndarray of numeric labels (N)
        num_classes: int
        images_per_class: dict[int, list[np.ndarray]] mapping class index to list of images
    """
    images = []
    labels = []
    images_per_class = {idx: [] for idx in range(len(class_names))}

    for c in sorted(os.listdir(directory)):
        folder_path = os.path.join(directory, c)
        if not os.path.isdir(folder_path):
            continue

        class_id = class_to_idx[c]

        for f in sorted(os.listdir(folder_path)):
            fpath = os.path.join(folder_path, f)
            if not os.path.isfile(fpath):
                continue
            try:
                img = cv2.imread(fpath)
                if img is None:
                    continue
                if BGR:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (pic_size, pic_size))
                images.append(img)
                labels.append(class_id)
                images_per_class[class_id].append(img)
            except Exception as e:
                print(f"Failed to load {fpath}: {e}")

    return np.array(images), np.array(labels), len(class_names), images_per_class


images, labels, num_classes, images_per_class = get_data_opencv_with_map(DATA_DIR, BGR=True)

print(f"Total images: {len(images)}")
print(f"Number of classes: {num_classes}")

for idx, imgs in images_per_class.items():
    print(f"Class {idx_to_class[idx]} ({idx}): {len(imgs)} images")


class SimpsonsDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert image to tensor and normalize
        image = torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.15, random_state=42, stratify=labels
)

train_dataset = SimpsonsDataset(X_train, y_train)
val_dataset = SimpsonsDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# training setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN4Conv(num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


num_epochs = 40
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f"Epoch {epoch + 1}: Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}")

torch.save(model.state_dict(), "simpsons_cnn4conv.pth")
print("Model weights saved to simpsons_cnn4conv.pth")
