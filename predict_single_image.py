import torch
from torchvision import transforms
from PIL import Image
from backend.model import MultiTaskModel
from facenet_pytorch import MTCNN

# === Constants ===
MODEL_PATH = "backend/models/multitask_model.pth"

# Skin type label map (based on training remapping)
skin_type_labels = {
    0: "Oily",
    1: "Normal",
    2: "Dry"
}
binary_labels = {0: "No", 1: "Yes"}

# === MTCNN ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)

def predict_with_remedy(image_path):
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Detect face and crop using MTCNN
    face_tensor = mtcnn(image)
    if face_tensor is None:
        return {"error": "❌ No face detected in the image. Please upload a clear selfie."}, None

    # Normalize and preprocess (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    face_tensor = transform(face_tensor).unsqueeze(0).to(device)

    # Load model
    model = MultiTaskModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Predict
    with torch.no_grad():
        skin_output, features_output = model(face_tensor)
        skin_pred = torch.argmax(skin_output, dim=1).item()
        feature_probs = torch.sigmoid(features_output).squeeze().cpu().numpy()
        feature_preds = (feature_probs > 0.5).astype(int)

    predictions = {
        "skin_type": skin_type_labels.get(skin_pred, "Unknown"),
        "acne": binary_labels[feature_preds[0]],
        "pimples": binary_labels[feature_preds[1]],
        "pigmentation": binary_labels[feature_preds[2]],
        "wrinkles": binary_labels[feature_preds[3]],
        "dark_circles": binary_labels[feature_preds[4]],
    }

    remedies = generate_remedies(predictions)
    return predictions, remedies

def generate_remedies(pred):
    remedies = {}

    if pred["skin_type"] == "Oily":
        remedies["skin_type"] = "Use salicylic acid-based cleansers. Avoid heavy moisturizers."
    elif pred["skin_type"] == "Dry":
        remedies["skin_type"] = "Apply hydrating moisturizers. Use mild cleansers."
    elif pred["skin_type"] == "Normal":
        remedies["skin_type"] = "Maintain a balanced skincare routine."

    if pred["acne"] == "Yes":
        remedies["acne"] = "Use benzoyl peroxide or salicylic acid. Avoid touching your face."

    if pred["pimples"] == "Yes":
        remedies["pimples"] = "Apply topical treatments like tea tree oil. Keep skin clean."

    if pred["pigmentation"] == "Yes":
        remedies["pigmentation"] = "Use products with niacinamide or vitamin C. Use sunscreen daily."

    if pred["wrinkles"] == "Yes":
        remedies["wrinkles"] = "Use retinoids, hydrate well. Apply sunscreen daily."

    if pred["dark_circles"] == "Yes":
        remedies["dark_circles"] = "Sleep well. Use caffeine-based eye creams or cold compresses."

    return remedies
