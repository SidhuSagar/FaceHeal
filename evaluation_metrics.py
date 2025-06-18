import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# =========================
# Paths
# =========================
CSV_PATH = 'dataset/final_labels.csv'
IMAGE_DIR = 'dataset/cropped'
MODEL_DIR = 'C:/Users/Dell/Desktop/faceheal_final/models'
MODEL_PATH = os.path.join(MODEL_DIR, 'multitask_model.pth')
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
# Evaluation Function
# =========================
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset & Dataloader
    dataset = SkinDataset(CSV_PATH, IMAGE_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Model
    model = MultiTaskModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Metrics storage
    skin_preds, skin_true = [], []
    feature_preds, feature_true = [], []

    with torch.no_grad():
        for images, skin_types, features in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            skin_types = skin_types.to(device)
            features = features.to(device)

            out_skin, out_features = model(images)

            # Skin type predictions
            skin_pred = torch.argmax(out_skin, dim=1)
            skin_preds.extend(skin_pred.cpu().numpy())
            skin_true.extend(skin_types.cpu().numpy())

            # Feature predictions (apply sigmoid and threshold at 0.5)
            feature_pred = (torch.sigmoid(out_features) > 0.5).float()
            feature_preds.extend(feature_pred.cpu().numpy())
            feature_true.extend(features.cpu().numpy())

    # Calculate metrics
    skin_accuracy = accuracy_score(skin_true, skin_preds)
    skin_precision = precision_score(skin_true, skin_preds, average='weighted')
    skin_recall = recall_score(skin_true, skin_preds, average='weighted')
    skin_f1 = f1_score(skin_true, skin_preds, average='weighted')

    feature_accuracy = accuracy_score(feature_true, feature_preds)
    feature_precision = precision_score(feature_true, feature_preds, average='weighted')
    feature_recall = recall_score(feature_true, feature_preds, average='weighted')
    feature_f1 = f1_score(feature_true, feature_preds, average='weighted')

    print("\nSkin Type Classification Metrics:")
    print(f"Accuracy: {skin_accuracy:.4f}")
    print(f"Precision: {skin_precision:.4f}")
    print(f"Recall: {skin_recall:.4f}")
    print(f"F1 Score: {skin_f1:.4f}")

    print("\nFeature Detection Metrics:")
    print(f"Accuracy: {feature_accuracy:.4f}")
    print(f"Precision: {feature_precision:.4f}")
    print(f"Recall: {feature_recall:.4f}")
    print(f"F1 Score: {feature_f1:.4f}")

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    evaluate()