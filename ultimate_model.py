from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K  # ✅ ADD THIS IMPORT
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import time
import json

print("🏆 ULTIMATE MODEL - MAXIMUM ACCURACY + FULL METRICS")
print("="*70)

# Start timer
start_time = time.time()

# Dataset path
DATASET_PATH = 'data'  # Using 'data' folder

if not os.path.exists(DATASET_PATH):
    print(f"❌ Dataset folder '{DATASET_PATH}' not found!")
    print("📁 Current directories:", os.listdir('.'))
    exit()

# Get class counts
normal_count = len(os.listdir(os.path.join(DATASET_PATH, 'train/NORMAL')))
pneumonia_count = len(os.listdir(
    os.path.join(DATASET_PATH, 'train/PNEUMONIA')))
print(f"📊 Training Data Balance:")
print(f"   Normal: {normal_count} images")
print(f"   Pneumonia: {pneumonia_count} images")
print(f"   Total: {normal_count + pneumonia_count} images")

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 20

# Advanced data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='constant',
    cval=0,
    brightness_range=[0.8, 1.2],
    validation_split=0.15
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15
)

# Load data
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    seed=42
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {val_generator.samples}")
print(f"✅ Test samples: {test_generator.samples}")

# Class weights
total_samples = normal_count + pneumonia_count
class_weights = {
    0: total_samples / (2 * normal_count),
    1: total_samples / (2 * pneumonia_count)
}
print(f"⚖️ Class Weights: {class_weights}")

# ============================================
# CUSTOM METRICS - FIXED VERSION
# ============================================


def precision_m(y_true, y_pred):
    """Calculate precision"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall_m(y_true, y_pred):
    """Calculate recall"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1_m(y_true, y_pred):
    """Calculate F1 score"""
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# ============================================
# BUILD MODEL
# ============================================


def create_ultimate_model():
    # Use ResNet50V2
    try:
        base_model = keras.applications.ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        print("✅ Using ResNet50V2 backbone")
        backbone_name = "ResNet50V2"
    except:
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        print("✅ Using ResNet50 backbone")
        backbone_name = "ResNet50"

    base_model.trainable = False

    # Build the model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    return model, base_model, backbone_name


print("🧠 Building Ultimate Model...")
model, base_model, backbone_name = create_ultimate_model()

# ============================================
# COMPILE WITH FIXED METRICS
# ============================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', precision_m, recall_m, f1_m]
)

print("📈 PHASE 1: Training classifier head (10 epochs)...")
history_phase1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=3, min_lr=0.0001)
    ],
    verbose=1
)

# ============================================
# PHASE 2: FINE-TUNE
# ============================================
print("\n📈 PHASE 2: Fine-tuning entire model (20 epochs)...")
base_model.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', precision_m, recall_m, f1_m]
)

history_phase2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=4, min_lr=0.00001),
        keras.callbacks.ModelCheckpoint(
            'best_ultimate_model.h5', save_best_only=True)
    ],
    verbose=1
)

# ============================================
# SAVE MODEL
# ============================================
model.save('pneumonia_model.h5')
print("\n✅ Ultimate model saved!")

# ============================================
# COMPREHENSIVE EVALUATION
# ============================================
print("\n" + "="*70)
print("🎯 COMPREHENSIVE MODEL EVALUATION")
print("="*70)

# Get predictions
y_true = test_generator.classes
y_pred_proba = model.predict(test_generator, verbose=1)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
y_pred_proba = y_pred_proba.flatten()

# Calculate metrics

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = recall

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"\n📊 TEST SET RESULTS:")
print("-"*50)
print(f"   🎯 Accuracy:     {accuracy:.4f} ({accuracy:.2%})")
print(f"   🎯 Precision:    {precision:.4f} ({precision:.2%})")
print(f"   🎯 Recall:       {recall:.4f} ({recall:.2%})")
print(f"   🎯 F1-Score:     {f1:.4f} ({f1:.2%})")
print(f"   🎯 Sensitivity:  {sensitivity:.4f} ({sensitivity:.2%})")
print(f"   🎯 Specificity:  {specificity:.4f} ({specificity:.2%})")
print(f"   🎯 AUC-ROC:      {roc_auc:.4f} ({roc_auc:.2%})")

print(f"\n📊 CONFUSION MATRIX:")
print("-"*50)
print(f"   True Negatives:  {tn}")
print(f"   False Positives: {fp}")
print(f"   False Negatives: {fn}")
print(f"   True Positives:  {tp}")

print(f"\n📊 CLASSIFICATION REPORT:")
print("-"*50)
print(classification_report(y_true, y_pred,
      target_names=['NORMAL', 'PNEUMONIA']))

# ============================================
# VISUALIZATIONS
# ============================================
print("\n📊 Generating visualizations...")

# 1. Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA'])
plt.title('Confusion Matrix - Ultimate Model', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_ultimate.png', dpi=300)
print("   ✅ Confusion matrix saved")

# 2. ROC Curve
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2,
        label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Receiver Operating Characteristic (ROC) Curve',
             fontsize=14, fontweight='bold')
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_ultimate.png', dpi=300)
print("   ✅ ROC curve saved")

# 3. Training History
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Combine histories


def combine_histories(h1, h2, metric):
    return h1.history[metric] + h2.history[metric]


epochs_phase1 = list(range(1, len(history_phase1.history['accuracy']) + 1))
epochs_phase2 = list(range(len(history_phase1.history['accuracy']) + 1,
                           len(history_phase1.history['accuracy']) + len(history_phase2.history['accuracy']) + 1))
epochs_combined = epochs_phase1 + epochs_phase2

accuracy_hist = combine_histories(history_phase1, history_phase2, 'accuracy')
val_accuracy_hist = combine_histories(
    history_phase1, history_phase2, 'val_accuracy')
loss_hist = combine_histories(history_phase1, history_phase2, 'loss')
val_loss_hist = combine_histories(history_phase1, history_phase2, 'val_loss')

axes[0, 0].plot(epochs_combined, accuracy_hist,
                'b-', label='Training', linewidth=2)
axes[0, 0].plot(epochs_combined, val_accuracy_hist,
                'r-', label='Validation', linewidth=2)
axes[0, 0].axvline(x=len(history_phase1.history['accuracy']),
                   color='gray', linestyle='--', alpha=0.5)
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epochs')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(epochs_combined, loss_hist, 'b-',
                label='Training', linewidth=2)
axes[0, 1].plot(epochs_combined, val_loss_hist, 'r-',
                label='Validation', linewidth=2)
axes[0, 1].axvline(x=len(history_phase1.history['accuracy']),
                   color='gray', linestyle='--', alpha=0.5)
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epochs')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Phase 2 metrics
if 'precision_m' in history_phase2.history:
    axes[1, 0].plot(history_phase2.history['precision_m'],
                    'b-', label='Training Precision', linewidth=2)
    axes[1, 0].plot(history_phase2.history['val_precision_m'],
                    'r-', label='Validation Precision', linewidth=2)
    axes[1, 0].set_title('Phase 2: Precision', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

if 'recall_m' in history_phase2.history:
    axes[1, 1].plot(history_phase2.history['recall_m'], 'b-',
                    label='Training Recall', linewidth=2)
    axes[1, 1].plot(history_phase2.history['val_recall_m'],
                    'r-', label='Validation Recall', linewidth=2)
    axes[1, 1].set_title('Phase 2: Recall', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ultimate_training_history.png', dpi=300)
print("   ✅ Training history saved")

# ============================================
# SAVE METRICS TO JSON
# ============================================
metrics_data = {
    'model_info': {
        'backbone': backbone_name,
        'epochs_phase1': EPOCHS_PHASE1,
        'epochs_phase2': EPOCHS_PHASE2,
        'batch_size': BATCH_SIZE,
        'image_size': IMG_SIZE
    },
    'dataset_info': {
        'normal_train': normal_count,
        'pneumonia_train': pneumonia_count,
        'total_train': normal_count + pneumonia_count,
        'test_samples': len(y_true)
    },
    'test_metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'auc_roc': float(roc_auc)
    },
    'confusion_matrix': {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
}

with open('training_results.json', 'w') as f:
    json.dump(metrics_data, f, indent=4)
print("   ✅ Metrics saved to training_results.json")

# ============================================
# FINAL SUMMARY
# ============================================
elapsed_time = time.time() - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)

print("\n" + "="*70)
print("🏆 ULTIMATE MODEL - FINAL SUMMARY")
print("="*70)

if accuracy >= 0.95:
    grade = "EXCELLENT 🏆"
elif accuracy >= 0.90:
    grade = "VERY GOOD ⭐"
elif accuracy >= 0.85:
    grade = "GOOD ✅"
elif accuracy >= 0.80:
    grade = "FAIR ⚠️"
else:
    grade = "NEEDS WORK 🔄"

print(f"\n📊 Performance Grade: {grade}")
print(f"\n📈 Key Metrics:")
print(f"   Accuracy:    {accuracy:.2%}")
print(f"   Precision:   {precision:.2%}")
print(f"   Recall:      {recall:.2%}")
print(f"   F1-Score:    {f1:.2%}")
print(f"   Specificity: {specificity:.2%}")
print(f"   AUC-ROC:     {roc_auc:.2%}")

print(f"\n📁 Files Generated:")
print(f"   - pneumonia_model.h5 (final model)")
print(f"   - best_ultimate_model.h5 (best checkpoint)")
print(f"   - confusion_matrix_ultimate.png")
print(f"   - roc_curve_ultimate.png")
print(f"   - ultimate_training_history.png")
print(f"   - training_results.json")

print(f"\n⏱️ Training Time: {hours}h {minutes}m")
print("\n🚀 Your Ultimate Model is ready for deployment!")
print("="*70)

# ============================================
# QUICK DEMO
# ============================================
print("\n🔬 Quick Demo on Sample Images:")
print("-"*50)


def test_sample_images():
    # Test normal images
    normal_dir = os.path.join(DATASET_PATH, 'test/NORMAL')
    if os.path.exists(normal_dir):
        normal_files = os.listdir(normal_dir)[:3]
        print("\n🟢 NORMAL IMAGES (should predict < 0.5):")
        for file in normal_files:
            img = keras.preprocessing.image.load_img(
                os.path.join(normal_dir, file), target_size=(224, 224))
            img_array = keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array, verbose=0)[0][0]
            status = "✅" if pred < 0.5 else "❌"
            print(f"   {status} {file[:35]}: {pred:.3f}")

    # Test pneumonia images
    pneumonia_dir = os.path.join(DATASET_PATH, 'test/PNEUMONIA')
    if os.path.exists(pneumonia_dir):
        pneumonia_files = os.listdir(pneumonia_dir)[:3]
        print("\n🔴 PNEUMONIA IMAGES (should predict > 0.5):")
        for file in pneumonia_files:
            img = keras.preprocessing.image.load_img(
                os.path.join(pneumonia_dir, file), target_size=(224, 224))
            img_array = keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array, verbose=0)[0][0]
            status = "✅" if pred > 0.5 else "❌"
            print(f"   {status} {file[:35]}: {pred:.3f}")


test_sample_images()
print("\n🎉 TRAINING COMPLETE! Your ultimate model with maximum accuracy is ready!")
