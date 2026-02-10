# your_project_root/app/app.py

import os
from flask import Flask, render_template, request, redirect, url_for, flash
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import numpy as np
import cv2 # For drawing on images
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/' # Folder to temporarily save uploaded images
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
app.secret_key = 'supersecretkey' # Needed for flash messages

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- 1. Configuration ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
CLASS_NAMES = ['dolphin', 'fish', 'lobster', 'octopus', 'sea_horse'] # MUST match order from train_models.py
MODEL_SAVE_DIR = '../models' # Relative path from app.py to models folder

# --- 2. Image Preprocessing for Inference ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. Model Definitions (MUST match train_models.py) ---
def load_mobilenetv2_model():
    model = models.mobilenet_v2(pretrained=False) # No pretrained weights for structure, load custom later
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, 'mobilenetv2_marine_classifier.pth'), map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

def load_efficientnetb0_model():
    model = models.efficientnet_b0(pretrained=False) # No pretrained weights for structure
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, 'efficientnetb0_marine_classifier.pth'), map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

class CustomCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def load_customcnn_model():
    model = CustomCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, 'customcnn_marine_classifier.pth'), map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

# --- 4. Load Models on App Startup ---
print("Loading models for inference...")
try:
    model_mobilenet = load_mobilenetv2_model()
    print("MobileNetV2 loaded.")
except FileNotFoundError:
    print("MobileNetV2 weights not found. Ensure train_models.py was run.")
    model_mobilenet = None

try:
    model_efficientnet = load_efficientnetb0_model()
    print("EfficientNet-B0 loaded.")
except FileNotFoundError:
    print("EfficientNet-B0 weights not found. Ensure train_models.py was run.")
    model_efficientnet = None

try:
    model_customcnn = load_customcnn_model()
    print("Custom CNN loaded.")
except FileNotFoundError:
    print("Custom CNN weights not found. Ensure train_models.py was run.")
    model_customcnn = None

all_models = {
    "MobileNetV2": model_mobilenet,
    "EfficientNet-B0": model_efficientnet,
    "Custom CNN": model_customcnn
}
all_models = {name: model for name, model in all_models.items() if model is not None} # Filter out failed loads
print(f"Active models: {list(all_models.keys())}")


# --- 5. Prediction Function ---
def predict_image(image_bytes, models_dict):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE) # Add batch dimension

    results = {}
    with torch.no_grad():
        for model_name, model in models_dict.items():
            if model is None: # Skip if model failed to load
                results[model_name] = {"prediction": "N/A", "confidence": "N/A", "error": "Model not loaded"}
                continue

            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
            
            results[model_name] = {
                "prediction": CLASS_NAMES[predicted_idx.item()],
                "confidence": f"{confidence.item()*100:.2f}%"
            }
    return results, img # Return original PIL image for drawing

# --- 6. Function to Draw Bounding Box (for classification visualization) ---
def draw_prediction_on_image(pil_img, prediction_text, confidence_text, model_name):
    # Convert PIL Image to OpenCV format
    img_cv = np.array(pil_img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Define bounding box around the whole image
    h, w, _ = img_cv.shape
    x1, y1, x2, y2 = 0, 0, w - 1, h - 1

    # Draw rectangle (bounding box)
    color = (0, 255, 0) # Green
    thickness = 2
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, thickness)

    # Put text for model name, prediction, and confidence
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (255, 255, 255) # White text

    # Position text
    text_y_offset = 30
    cv2.putText(img_cv, f"Model: {model_name}", (x1 + 10, y1 + text_y_offset), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(img_cv, f"Class: {prediction_text}", (x1 + 10, y1 + text_y_offset + 30), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(img_cv, f"Conf: {confidence_text}", (x1 + 10, y1 + text_y_offset + 60), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Convert back to PIL Image and then to base64
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    img_pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# --- 7. Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            try:
                img_bytes = file.read()
                results, original_img_pil = predict_image(img_bytes, all_models)

                processed_images_base64 = {}
                for model_name, res in results.items():
                    if "error" in res:
                        processed_images_base64[model_name] = f"Error: {res['error']}"
                    else:
                        processed_images_base64[model_name] = draw_prediction_on_image(
                            original_img_pil, res['prediction'], res['confidence'], model_name
                        )

                # Convert original image to base64 for display
                original_img_buffered = io.BytesIO()
                original_img_pil.save(original_img_buffered, format="JPEG")
                original_img_base64 = base64.b64encode(original_img_buffered.getvalue()).decode('utf-8')

                return render_template('index.html', 
                                       results=results, 
                                       original_img=original_img_base64,
                                       processed_images=processed_images_base64)
            except Exception as e:
                flash(f'Error processing image: {e}')
                return redirect(request.url)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) # Set debug=False for production