import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests from frontend

# Paths
MODEL_PATH = 'C:/Users/Dell/Desktop/faceheal_final/models/multitask_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Class (corrected from train.py)
class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        base = models.resnet18(pretrained=False)
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

# Load Model
model = MultiTaskModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Image Transform (same as train.py)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Prediction Function
def predict_image(image):
    image = Image.open(image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        skin_type_out, features_out = model(image)
        skin_type_probs = torch.softmax(skin_type_out, dim=1).cpu().numpy()[0]
        features_probs = torch.sigmoid(features_out).cpu().numpy()[0]
    
    # Map skin types back to original labels
    skin_type_map = {0: 'Type 2', 1: 'Type 3', 2: 'Type 4'}
    skin_type_pred = skin_type_map[skin_type_probs.argmax()]
    
    # Map feature predictions (threshold at 0.5 for binary classification)
    feature_labels = ['Acne', 'Pimples', 'Pigmentation', 'Wrinkles', 'Dark Circles']
    features_pred = {label: float(prob) for label, prob in zip(feature_labels, features_probs)}
    
    return {'skin_type': skin_type_pred, 'features': features_pred}

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image = request.files['image']
    try:
        result = predict_image(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)