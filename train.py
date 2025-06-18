import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm

# =========================
# Paths
# =========================
CSV_PATH = 'dataset/final_labels.csv'
IMAGE_DIR = 'dataset/cropped'
MODEL_DIR = 'C:/Users/Dell/Desktop/faceheal_final/models'
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# Dataset Class
# =========================
class SkinDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Remap skin_type from {2,3,4} to {0,1,2}
        skin_type_map = {2: 0, 3: 1, 4: 2}
        skin_type = skin_type_map[int(row['skin_type'])]

        # Binary labels
        acne = int(row['acne'])
        pimples = int(row['pimples'])
        pigmentation = int(row['pigmentation'])
        wrinkles = int(row['wrinkles'])
        dark_circles = int(row['dark_circles'])

        multi_labels = torch.tensor(
            [acne, pimples, pigmentation, wrinkles, dark_circles],
            dtype=torch.float32
        )

        return image, skin_type, multi_labels

# =========================
# Model Class (ResNet18 with 2 Heads)
# =========================
class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        base = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # Remove FC layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc_skin_type = nn.Linear(512, 3)   # 3 skin types
        self.fc_features = nn.Linear(512, 5)    # 5 binary outputs

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flatten(x)

        skin_type_out = self.fc_skin_type(x)
        features_out = self.fc_features(x)

        return skin_type_out, features_out

# =========================
# Train Function
# =========================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset & Dataloader
    dataset = SkinDataset(CSV_PATH, IMAGE_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = MultiTaskModel().to(device)

    # Loss Functions
    loss_skin = nn.CrossEntropyLoss()
    loss_features = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training Loop
    model.train()
    for epoch in range(10):
        running_loss = 0.0
        for images, skin_types, features in tqdm(dataloader, desc=f"Epoch {epoch+1}/10"):
            images = images.to(device)
            skin_types = skin_types.to(device)
            features = features.to(device)

            optimizer.zero_grad()
            out_skin, out_features = model(images)

            loss1 = loss_skin(out_skin, skin_types)
            loss2 = loss_features(out_features, features)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {running_loss / len(dataloader):.4f}")

    # Save Model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'multitask_model.pth'))
    print("[✅] Model training complete and saved.")

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    train()  