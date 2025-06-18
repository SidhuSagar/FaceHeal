import os
import io
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN

from model_utils import MultiTaskModel  # Ensure this file defines the model architecture

# ─────────────────────────────────────────────────────────────
# ✅ App Setup and CORS
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# ✅ Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
root_project_dir = os.path.normpath(os.path.join(current_dir, ".."))

# Mount static files (e.g., HTML/CSS/JS)
app.mount("/static", StaticFiles(directory=root_project_dir), name="static")

# Mount images folder (to serve image-1.jpg, etc.)
images_path = os.path.join(current_dir, "images")
app.mount("/images", StaticFiles(directory=images_path), name="images")

# ─────────────────────────────────────────────────────────────
# ✅ Device + Model Load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load face detector
mtcnn = MTCNN(keep_all=False, device=device)

# Load skin analysis model
MODEL_PATH = os.path.join(root_project_dir, "models", "multitask_model.pth")
MODEL_PATH = os.path.normpath(MODEL_PATH)

try:
    model = MultiTaskModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
except FileNotFoundError:
    print(f"❌ Model file not found at {MODEL_PATH}")
    exit()

# ─────────────────────────────────────────────────────────────
# ✅ Labels
skin_type_labels = {0: "Oily", 1: "Normal", 2: "Dry"}
binary_labels = {0: "No", 1: "Yes"}

# ─────────────────────────────────────────────────────────────
# ✅ Remedies Function
def generate_remedies(pred):
    remedies = {}

    # Individual Skin Type Remedies
    if pred["skin_type"] == "Oily":
        remedies["skin_type"] = (
            "Use salicylic acid or tea tree oil cleansers to unclog pores. "
            "Avoid heavy moisturizers; opt for oil-free, non-comedogenic products. "
            "Clay masks (2x/week) help absorb excess oil."
        )
    elif pred["skin_type"] == "Dry":
        remedies["skin_type"] = (
            "Use gentle, hydrating cleansers. Apply thick moisturizers with hyaluronic acid. "
            "Avoid alcohol-based products. Use overnight hydrating masks twice a week."
        )
    elif pred["skin_type"] == "Normal":
        remedies["skin_type"] = (
            "Maintain a balanced skincare routine. Cleanse twice daily, exfoliate weekly, "
            "and hydrate with a light moisturizer. Don’t skip sunscreen."
        )

    # Individual Conditions
    if pred["acne"] == "Yes":
        remedies["acne"] = (
            "Use products with benzoyl peroxide or salicylic acid. "
            "Avoid scrubbing. Keep pillowcases clean. "
            "Consider niacinamide serum and reduce sugar/dairy intake."
        )
    if pred["pimples"] == "Yes":
        remedies["pimples"] = (
            "Apply tea tree oil or pimple patches. "
            "Use gentle spot treatments with sulfur or salicylic acid. "
            "Don’t pop pimples to prevent scarring."
        )
    if pred["wrinkles"] == "Yes":
        remedies["wrinkles"] = (
            "Use retinol or bakuchiol at night. Apply sunscreen daily to prevent photoaging. "
            "Incorporate peptides and antioxidants. Stay hydrated and sleep on your back."
        )
    if pred["pigmentation"] == "Yes":
        remedies["pigmentation"] = (
            "Use products with vitamin C, kojic acid, or licorice extract. "
            "Exfoliate with AHAs weekly. Always wear broad-spectrum sunscreen."
        )
    if pred["dark_circles"] == "Yes":
        remedies["dark_circles"] = (
            "Sleep at least 7 hours. Apply cold tea bags or caffeine serums. "
            "Use eye creams with niacinamide and retinol. Stay hydrated."
        )

    # Combination Remedies
    skin = pred["skin_type"]
    if skin == "Oily" and pred["acne"] == "Yes":
        remedies["combo_oily_acne"] = (
            "Use a salicylic acid-based foaming cleanser. Add niacinamide serum to reduce oil and inflammation. "
            "Avoid coconut oil or heavy creams. Use blotting paper during the day."
        )
    if skin == "Dry" and pred["wrinkles"] == "Yes":
        remedies["combo_dry_wrinkles"] = (
            "Apply ceramide-rich creams to restore moisture. Use bakuchiol or low-strength retinol to treat fine lines. "
            "Avoid over-exfoliation and consider a humidifier at night."
        )
    if pred["pigmentation"] == "Yes" and pred["dark_circles"] == "Yes":
        remedies["combo_pigmentation_darkcircles"] = (
            "Use a vitamin C serum in the morning and niacinamide at night. Apply sunscreen daily. "
            "Use color-correcting concealers and caffeine eye creams to reduce both issues."
        )
    if pred["wrinkles"] == "Yes" and pred["acne"] == "Yes":
        remedies["combo_wrinkles_acne"] = (
            "Alternate between salicylic acid (for acne) and retinol (for wrinkles). "
            "Use a calming moisturizer to balance both concerns. Avoid overuse of actives."
        )

    return remedies 


# ─────────────────────────────────────────────────────────────
# ✅ Frontend Route
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_file_path = os.path.join(root_project_dir, "frontpage.html")
    try:
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="❌ frontpage.html not found at expected path.", status_code=404)

# ─────────────────────────────────────────────────────────────
# ✅ Predict Endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    boxes, probs = mtcnn.detect(image)

    if boxes is None or len(boxes) == 0 or probs[0] < 0.90:
        return JSONResponse(status_code=400, content={"error": "❌ No confident face detected in the image."})

    face_tensor = mtcnn(image)
    if face_tensor is None:
        return JSONResponse(status_code=400, content={"error": "❌ Face extraction failed."})

    if isinstance(face_tensor, list) and len(face_tensor) > 0:
        face_tensor = face_tensor[0]
    elif not isinstance(face_tensor, torch.Tensor):
        return JSONResponse(status_code=500, content={"error": "Unexpected MTCNN output type."})

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    face_tensor = transform(face_tensor).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        skin_out, features_out = model(face_tensor)
        skin_pred = torch.argmax(skin_out, dim=1).item()
        feature_probs = torch.sigmoid(features_out).squeeze().cpu().numpy()
        feature_preds = (feature_probs > 0.5).astype(int)

    predictions = {
        "skin_type": skin_type_labels.get(skin_pred, "Unknown"),
        "acne": binary_labels[feature_preds[0]],
        "pimples": binary_labels[feature_preds[1]],
        "wrinkles": binary_labels[feature_preds[2]],
        "pigmentation": binary_labels[feature_preds[3]],
        "dark_circles": binary_labels[feature_preds[4]],
    }

    remedies = generate_remedies(predictions)

    return {
        "predictions": predictions,
        "remedies": remedies
    }