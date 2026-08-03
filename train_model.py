import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import json
import os

# Load class names
with open('class_names.json') as f:
    class_names = json.load(f)

dataset_dir = 'organized_datasets'
if not os.path.exists(dataset_dir):
    print(f"⚠️ Warning: '{dataset_dir}' directory not found. Please place your training image folders inside '{dataset_dir}/'.")

# Image generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

if os.path.exists(dataset_dir):
    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Build model using MobileNetV2 Transfer Learning
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(train_gen.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model & record history
    print("🚀 Starting MobileNetV2 model training...")
    history = model.fit(train_gen, validation_data=val_gen, epochs=10)

    # Save model
    model.save('crop_disease_model.h5')
    val_acc = float(history.history['val_accuracy'][-1])

    # Save accuracy stats to model_accuracy.json
    with open('model_accuracy.json', 'w') as f:
        json.dump({'accuracy': val_acc, 'epochs': 10, 'classes': train_gen.num_classes}, f, indent=2)

    print(f"✅ Training Complete! Model saved to crop_disease_model.h5 with Final Validation Accuracy: {val_acc * 100:.2f}%")
else:
    print("❌ Cannot train model: 'organized_datasets' folder is missing.")