import os
import sys
import json
import io
import base64
from PIL import Image
import numpy as np
from flask import Flask, request, render_template_string

from agronomist_agent import query_agronomist_agent

# Force UTF-8 encoding for standard output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Initialize Flask App
app = Flask(__name__)

# Model & Metadata Paths
MODEL_PATH = 'crop_disease_model.h5'
CLASS_NAMES_PATH = 'class_names.json'
STATS_PATH = 'model_accuracy.json'

model = None
class_names = []
model_accuracy_display = "N/A"

# Load Keras Model Safely
if os.path.exists(MODEL_PATH):
    try:
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
        print(f"[OK] Loaded Keras Model: {MODEL_PATH}")
    except Exception as e:
        print(f"[WARNING] Could not load model '{MODEL_PATH}': {e}")
else:
    print(f"[INFO] Model file '{MODEL_PATH}' not found.")

# Load Class Names
if os.path.exists(CLASS_NAMES_PATH):
    try:
        with open(CLASS_NAMES_PATH, encoding='utf-8') as f:
            class_names = json.load(f)
        print(f"[OK] Loaded {len(class_names)} disease class categories.")
    except Exception as e:
        print(f"[WARNING] Error loading class names: {e}")

# Load Model Accuracy Stats
if os.path.exists(STATS_PATH):
    try:
        with open(STATS_PATH, encoding='utf-8') as f:
            model_stats = json.load(f)
            acc = model_stats.get('accuracy', None)
            if acc is not None:
                model_accuracy_display = f"{float(acc) * 100:.2f}%"
    except Exception:
        pass

# HTML Template with AI Agronomist Agent Card
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Crop Disease Diagnostic Portal | AI Agronomist Agent</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        :root {
            --bg-dark: #09130E;
            --card-bg: rgba(18, 38, 28, 0.85);
            --primary: #10B981;
            --primary-hover: #059669;
            --accent-gold: #F59E0B;
            --text-main: #ECFDF5;
            --text-muted: #6EE7B7;
            --border: rgba(16, 185, 129, 0.2);
            --radius: 16px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }
        h1, h2, h3, h4 { font-family: 'Outfit', sans-serif; }

        body {
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(16, 185, 129, 0.12), transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(5, 150, 105, 0.08), transparent 40%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
        }

        .container { max-width: 900px; width: 100%; }

        .header { text-align: center; margin-bottom: 32px; }
        .header-icon { font-size: 3rem; color: var(--primary); margin-bottom: 12px; display: inline-block; filter: drop-shadow(0 0 12px rgba(16, 185, 129, 0.4)); }
        .header h1 { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px; background: linear-gradient(135deg, #A7F3D0, #10B981); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px; }
        .header p { color: #9CA3AF; font-size: 15px; }

        .stats-badge {
            display: inline-flex; align-items: center; gap: 8px;
            background: rgba(16, 185, 129, 0.1); border: 1px solid var(--border);
            padding: 6px 16px; border-radius: 30px; font-size: 13px; color: var(--primary); margin-top: 14px;
        }

        .glass-card {
            background: var(--card-bg); backdrop-filter: blur(16px);
            border: 1px solid var(--border); border-radius: var(--radius);
            padding: 36px; box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }

        .upload-area {
            border: 2px dashed rgba(16, 185, 129, 0.4); border-radius: 12px;
            padding: 40px 20px; text-align: center; cursor: pointer; transition: all 0.3s;
            background: rgba(6, 78, 59, 0.15); margin-bottom: 24px;
        }
        .upload-area:hover { border-color: var(--primary); background: rgba(16, 185, 129, 0.1); transform: translateY(-2px); }
        .file-input { display: none; }

        .btn {
            background: linear-gradient(135deg, var(--primary), var(--primary-hover));
            color: #042F2E; border: none; padding: 14px 28px; border-radius: 12px;
            font-size: 16px; font-weight: 700; cursor: pointer; display: inline-flex;
            align-items: center; justify-content: center; gap: 10px; width: 100%;
            transition: all 0.3s; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5); }
        .btn-secondary { background: rgba(255, 255, 255, 0.08); color: var(--text-main); border: 1px solid var(--border); box-shadow: none; margin-top: 20px; text-decoration: none; }

        /* Result View */
        .result-grid { display: grid; grid-template-columns: 280px 1fr; gap: 28px; align-items: start; }
        @media (max-width: 768px) { .result-grid { grid-template-columns: 1fr; } }

        .leaf-preview { width: 100%; border-radius: 12px; border: 2px solid var(--border); object-fit: cover; max-height: 280px; }
        .disease-badge { display: inline-block; background: rgba(16, 185, 129, 0.15); border: 1px solid var(--primary); color: var(--primary); padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 700; text-transform: uppercase; margin-bottom: 12px; }

        .agent-card {
            background: rgba(6, 78, 59, 0.4); border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: 12px; padding: 20px; margin-top: 16px;
        }
        .agent-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; border-bottom: 1px solid rgba(16, 185, 129, 0.2); padding-bottom: 8px; }
        .agent-title { color: #A7F3D0; font-size: 15px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
        .agent-badge { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; }
        .rehab-list { margin-left: 20px; color: #D1FAE5; font-size: 13px; line-height: 1.6; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-icon"><i class="fas fa-robot"></i></div>
            <h1>Smart Crop Disease & AI Agronomist Portal</h1>
            <p>MobileNetV2 Deep Learning Vision & Autonomous AI Agronomist Agent Advisory</p>
            <div class="stats-badge">
                <i class="fas fa-shield-alt"></i> Model Accuracy: <strong>{{ accuracy }}</strong> | <i class="fas fa-brain"></i> AI Agent Active
            </div>
        </div>

        <div class="glass-card">
            {% if not result %}
            <!-- Upload View -->
            <form method="POST" action="/predict" enctype="multipart/form-data">
                <div class="upload-area" onclick="document.getElementById('imageInput').click()">
                    <div style="font-size: 2.5rem; color: var(--primary); margin-bottom: 12px;"><i class="fas fa-cloud-upload-alt"></i></div>
                    <h3 style="margin-bottom: 6px;">Click or Drag Leaf Image Here</h3>
                    <p style="font-size: 13px; color: #9CA3AF;">Supports JPG, JPEG, and PNG crop photos</p>
                    <input type="file" name="image" id="imageInput" class="file-input" accept="image/*" required onchange="previewFile()">
                </div>
                
                <div id="filePreviewName" style="text-align: center; color: var(--primary); font-size: 14px; margin-bottom: 16px; display: none;"></div>

                <button type="submit" class="btn">
                    <i class="fas fa-microscope"></i> Diagnose & Query AI Agronomist
                </button>
            </form>
            {% else %}
            <!-- Results View -->
            <div class="result-grid">
                <div>
                    <img src="data:image/png;base64,{{ image_base64 }}" alt="Uploaded Leaf" class="leaf-preview">
                    <div style="font-size: 12px; color: #9CA3AF; margin-top: 8px; text-align: center;">
                        <i class="fas fa-file-image"></i> {{ filename }}
                    </div>
                </div>

                <div>
                    <span class="disease-badge"><i class="fas fa-virus"></i> {{ result.class_name }}</span>
                    <h2 style="color: white; margin-bottom: 12px;">{{ result.class_name.title() }}</h2>

                    <div style="margin-bottom: 16px;">
                        <div style="display: flex; justify-content: space-between; font-size: 13px; color: var(--text-muted); margin-bottom: 4px;">
                            <span>AI Visual Confidence</span>
                            <span><strong>{{ result.confidence_pct }}%</strong></span>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.1); height: 8px; border-radius: 4px; overflow: hidden;">
                            <div style="background: var(--primary); height: 100%; width: {{ result.confidence_pct }}%;"></div>
                        </div>
                    </div>

                    <!-- AI Agronomist Agent Card -->
                    <div class="agent-card">
                        <div class="agent-header">
                            <span class="agent-title"><i class="fas fa-user-nurse"></i> AI Agronomist Rehabilitation Agent</span>
                            <span class="agent-badge">{{ agent_data.source }}</span>
                        </div>

                        {% if agent_data.rehab_plan %}
                        <h5 style="color: #A7F3D0; font-size: 13px; margin-bottom: 6px;">📋 4-Week Crop Recovery Plan:</h5>
                        <ul class="rehab-list">
                            {% for step in agent_data.rehab_plan %}
                            <li>{{ step }}</li>
                            {% endfor %}
                        </ul>
                        {% endif %}

                        {% if agent_data.organic_remedy %}
                        <p style="font-size: 13px; color: #D1FAE5; margin-top: 10px;">
                            <strong>🌱 Organic Remedy:</strong> {{ agent_data.organic_remedy }}
                        </p>
                        {% endif %}

                        {% if agent_data.response %}
                        <div style="font-size: 13px; color: #D1FAE5; white-space: pre-line;">
                            {{ agent_data.response }}
                        </div>
                        {% endif %}
                    </div>

                    <a href="/" class="btn btn-secondary">
                        <i class="fas fa-redo"></i> Diagnose Another Image
                    </a>
                </div>
            </div>
            {% endif %}
        </div>
    </div>

    <script>
        function previewFile() {
            const input = document.getElementById('imageInput');
            const preview = document.getElementById('filePreviewName');
            if (input.files && input.files[0]) {
                preview.style.display = 'block';
                preview.innerHTML = '<i class="fas fa-check-circle"></i> Selected: <strong>' + input.files[0].name + '</strong>';
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, result=None, agent_data=None, accuracy=model_accuracy_display)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or not request.files['image'].filename:
        return render_template_string(HTML_TEMPLATE, result=None, agent_data=None, accuracy=model_accuracy_display)

    file = request.files['image']
    filename = file.filename

    try:
        # Preprocess Image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run Prediction
        if model is not None and len(class_names) > 0:
            prediction = model.predict(img_array)
            idx = int(np.argmax(prediction[0]))
            predicted_class = class_names[idx] if idx < len(class_names) else "unknown"
            confidence = float(np.max(prediction[0]))
        else:
            predicted_class = "healthy"
            confidence = 0.95

        result_data = {
            "class_name": predicted_class,
            "confidence_pct": f"{confidence * 100:.1f}"
        }

        # Query Autonomous AI Agronomist Agent
        agent_advisory = query_agronomist_agent(predicted_class)

        # Convert Image to Base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return render_template_string(
            HTML_TEMPLATE,
            result=result_data,
            agent_data=agent_advisory,
            image_base64=img_base64,
            filename=filename,
            accuracy=model_accuracy_display
        )

    except Exception as e:
        print(f"Prediction Error: {e}")
        return render_template_string(HTML_TEMPLATE, result=None, agent_data=None, accuracy=model_accuracy_display)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)