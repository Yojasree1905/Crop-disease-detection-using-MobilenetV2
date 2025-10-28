from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model
import numpy as np
import json
from PIL import Image
import io
import base64

app = Flask(__name__)
model = load_model('crop_disease_model.h5')

# Load class names
with open('class_names.json') as f:
    class_names = json.load(f)

# Load model accuracy
try:
    with open('model_accuracy.json') as f:
        model_stats = json.load(f)
        model_accuracy = model_stats.get('accuracy', 'N/A')
except:
    model_accuracy = 'N/A'

# Advisory knowledge base
advisory = {
    "anthracnose": "Remove infected leaves, apply copper-based fungicide.",
    "bacterial blight": "Use resistant varieties, avoid overhead irrigation.",
    "brown spot": "Apply fungicide, improve air circulation.",
    "fall armyworm": "Use pheromone traps, apply neem-based pesticide.",
    "grasshopper": "Manual removal, use biological control agents.",
    "green mite": "Spray miticide, prune affected areas.",
    "gummosis": "Improve drainage, apply fungicide to wounds.",
    "healthy": "No action needed. Keep monitoring.",
    "mosaic": "Remove infected plants, control aphids.",
    "red rust": "Apply copper fungicide, remove infected leaves.",
    "streak virus": "Use virus-free seeds, control insect vectors.",
    "verticillium wilt": "Rotate crops, remove infected plants."
}

@app.route('/')
def home():
    return render_template_string('''
        <h2>Smart Crop Disease Detector</h2>
        <form method="POST" action="/predict" enctype="multipart/form-data">
            <input type="file" name="image" required>
            <input type="submit" value="Diagnose">
        </form>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filename = file.filename
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    advice = advisory.get(predicted_class, "No advisory available.")

    # Safely format model accuracy
    try:
        accuracy_display = f"{float(model_accuracy) * 100:.2f}%"
    except ValueError:
        accuracy_display = "N/A"

    # Convert image to base64 for preview
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return render_template_string(f'''
        <h2>Prediction Result</h2>
        <p><strong>Uploaded File:</strong> {filename}</p>
        <img src="data:image/png;base64,{img_base64}" alt="Uploaded Leaf" width="300"><br><br>
        <p><strong>Disease:</strong> {predicted_class}</p>
        <p><strong>Confidence:</strong> {confidence:.2f}</p>
        <p><strong>Advisory:</strong> {advice}</p>
        <p><strong>Model Accuracy:</strong> {accuracy_display}</p>
        <a href="/">Try another image</a>
    ''')

if __name__ == '__main__':
    app.run(debug=True)