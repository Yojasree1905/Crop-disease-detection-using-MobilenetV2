import os
import sys
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model and class names
model = load_model('crop_disease_model.h5')
with open('class_names.json') as f:
    class_names = json.load(f)

def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        return predicted_class, confidence
    except Exception as e:
        return f"Error processing {img_path}: {e}", None

def predict_folder(folder_path):
    supported_exts = ('.jpg', '.jpeg', '.png')
    results = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(supported_exts):
            full_path = os.path.join(folder_path, fname)
            label, score = predict_image(full_path)
            results.append((fname, label, score))
    return results

if __name__ == "__main__":
    # Get path from command line or prompt
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter path to image or folder: ").strip()

    if not os.path.exists(path):
        print(f"❌ File or folder not found: {path}")
    elif os.path.isfile(path):
        label, score = predict_image(path)
        if score is not None:
            print(f"\n✅ Prediction for {os.path.basename(path)}: {label} ({score:.2f} confidence)")
        else:
            print(f"\n❌ {label}")
    elif os.path.isdir(path):
        print(f"\n📂 Running batch prediction on folder: {path}\n")
        results = predict_folder(path)
        for fname, label, score in results:
            if score is not None:
                print(f"✅ {fname}: {label} ({score:.2f})")
            else:
                print(f"❌ {fname}: {label}")
    else:
        print("❌ Invalid path type.")