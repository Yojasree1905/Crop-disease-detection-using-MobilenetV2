import os
import sys
import json
import numpy as np

# Force UTF-8 encoding for standard output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

MODEL_PATH = 'crop_disease_model.h5'
CLASS_NAMES_PATH = 'class_names.json'

model = None
class_names = []

# Load model and class names safely
if os.path.exists(MODEL_PATH):
    try:
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"[WARNING] Error loading model: {e}")

if os.path.exists(CLASS_NAMES_PATH):
    try:
        with open(CLASS_NAMES_PATH, encoding='utf-8') as f:
            class_names = json.load(f)
    except Exception as e:
        print(f"[WARNING] Error loading class names: {e}")

def predict_image(img_path):
    """Predict crop disease for a single image file"""
    if model is None or not class_names:
        return "Model or class names not loaded properly", None

    try:
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        idx = int(np.argmax(prediction[0]))
        predicted_class = class_names[idx] if idx < len(class_names) else "Unknown"
        confidence = float(np.max(prediction[0]))
        return predicted_class, confidence
    except Exception as e:
        return f"Error processing image: {e}", None

def predict_folder(folder_path):
    """Run batch crop disease predictions for all supported images in a folder"""
    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    results = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith(supported_exts):
            full_path = os.path.join(folder_path, fname)
            label, score = predict_image(full_path)
            results.append((fname, label, score))
    return results

if __name__ == "__main__":
    print("\nMobileNetV2 Crop Disease Diagnostic CLI Tool\n")

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter path to leaf image or directory: ").strip()

    if not os.path.exists(path):
        print(f"[ERROR] File or folder not found: {path}")
    elif os.path.isfile(path):
        label, score = predict_image(path)
        if score is not None:
            print(f"[OK] Image: {os.path.basename(path)}")
            print(f"     Diagnosis: {label.upper()}")
            print(f"     Confidence: {score * 100:.2f}%\n")
        else:
            print(f"[ERROR] {label}\n")
    elif os.path.isdir(path):
        print(f"Running batch prediction on folder: {path}\n")
        results = predict_folder(path)
        print(f"{'Filename':<30} | {'Predicted Disease':<22} | {'Confidence'}")
        print("-" * 70)
        for fname, label, score in results:
            if score is not None:
                print(f"{fname:<30} | {label:<22} | {score * 100:.2f}%")
            else:
                print(f"{fname:<30} | {label:<22} | N/A")
        print()
    else:
        print("[ERROR] Invalid path specified.")