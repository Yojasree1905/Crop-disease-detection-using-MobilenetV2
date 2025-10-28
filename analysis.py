import os
import json
import time
import psutil
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent Tkinter errors in headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Optional: GPU memory logging
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        gpu_info = tf.config.experimental.get_memory_info('GPU:0')
        print(f"🧠 GPU Memory - Used: {gpu_info['current'] / 1024**2:.2f} MB")
except Exception as e:
    print("⚠️ GPU memory info not available")

# Load model and class names
model = load_model('crop_disease_model.h5')
with open('class_names.json') as f:
    class_names = json.load(f)

# Load validation data
target_size = (224, 224)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_gen = datagen.flow_from_directory(
    'organized_datasets',
    target_size=target_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Predict
start_time = time.time()
predictions = model.predict(val_gen)
end_time = time.time()
y_pred = np.argmax(predictions, axis=1)
y_true = val_gen.classes

print(f"\n⏱️ Inference Time: {end_time - start_time:.2f} seconds")
print(f"🧠 RAM Used: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(val_gen.class_indices.keys()),
            yticklabels=list(val_gen.class_indices.keys()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Classification Report
report = classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys()))
print("\n📊 Classification Report:\n")
print(report)

# Ablation Study
image_sizes = [(128, 128), (224, 224), (256, 256)]
ablation_results = []

for size in image_sizes:
    print(f"\n🔍 Ablation: Training with image size {size}")
    train_gen = datagen.flow_from_directory(
        'organized_datasets',
        target_size=size,
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    val_gen_ablation = datagen.flow_from_directory(
        'organized_datasets',
        target_size=size,
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(size[0], size[1], 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(train_gen.num_classes, activation='softmax')(x)
    ablation_model = Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False

    ablation_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    start = time.time()
    history = ablation_model.fit(train_gen, validation_data=val_gen_ablation, epochs=5, verbose=0)
    duration = time.time() - start
    val_acc = history.history['val_accuracy'][-1]
    ablation_results.append((f"{size[0]}x{size[1]}", round(val_acc, 4), round(duration, 2)))

# Plot Ablation Results
sizes = [r[0] for r in ablation_results]
accs = [r[1] for r in ablation_results]

plt.figure(figsize=(8, 6))
bars = plt.bar(sizes, accs, color='skyblue')
plt.title("Ablation Study: Image Size vs Validation Accuracy")
plt.xlabel("Image Size")
plt.ylabel("Validation Accuracy")
plt.ylim(0.85, 0.92)
for bar, (_, acc, dur) in zip(bars, ablation_results):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f"{acc:.4f}\n{dur}s", ha='center')
plt.tight_layout()
plt.savefig("ablation_study_chart.png")
plt.close()

# Model Comparison
models_to_test = {
    "MobileNetV2": MobileNetV2,
    "ResNet50": ResNet50,
    "EfficientNetB0": EfficientNetB0
}
comparison_results = []

for name, model_fn in models_to_test.items():
    print(f"\n🔍 Comparing: {name}")
    base_model = model_fn(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(train_gen.num_classes, activation='softmax')(x)
    comp_model = Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False

    comp_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    start = time.time()
    history = comp_model.fit(train_gen, validation_data=val_gen, epochs=5, verbose=0)
    duration = time.time() - start
    val_acc = history.history['val_accuracy'][-1]
    comparison_results.append((name, round(val_acc, 4), round(duration, 2)))

# Plot Model Comparison
names = [r[0] for r in comparison_results]
accs = [r[1] for r in comparison_results]

plt.figure(figsize=(8, 6))
bars = plt.bar(names, accs, color='lightgreen')
plt.title("Model Comparison: Validation Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.85, 0.95)
for bar, (_, acc, dur) in zip(bars, comparison_results):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f"{acc:.4f}\n{dur}s", ha='center')
plt.tight_layout()
plt.savefig("model_comparison_chart.png")
plt.close()