import os
import random
import json
import csv
import numpy as np

# 1. Set reproducibility seeds
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

def train_custom_cnn():
    processed_dir = "processed_data"
    train_dir = os.path.join(processed_dir, "train")
    val_dir = os.path.join(processed_dir, "val")
    test_dir = os.path.join(processed_dir, "test")

    # Error handling for missing directories
    for path, name in [(train_dir, "Train"), (val_dir, "Validation"), (test_dir, "Test")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} directory does not exist at '{path}'. Please ensure dataset is prepared.")
        for cls in ["Fire", "Non-Fire"]:
            cls_path = os.path.join(path, cls)
            if not os.path.exists(cls_path):
                raise FileNotFoundError(f"Class directory '{cls}' missing in {name} path: '{cls_path}'")

    print("=" * 60)
    print("FOREST GUARDIAN AI — CUSTOM CNN TRAINING & EVALUATION")
    print("=" * 60)

    # 1. Dataset Counts
    def count_images(dir_path):
        fire_cnt = len(os.listdir(os.path.join(dir_path, "Fire")))
        non_fire_cnt = len(os.listdir(os.path.join(dir_path, "Non-Fire")))
        return fire_cnt, non_fire_cnt

    train_fire, train_non_fire = count_images(train_dir)
    val_fire, val_non_fire = count_images(val_dir)
    test_fire, test_non_fire = count_images(test_dir)

    print("\nDataset Image Counts:")
    print(f"  Train: Fire = {train_fire}, Non-Fire = {train_non_fire} (Total: {train_fire + train_non_fire})")
    print(f"  Val:   Fire = {val_fire}, Non-Fire = {val_non_fire} (Total: {val_fire + val_non_fire})")
    print(f"  Test:  Fire = {test_fire}, Non-Fire = {test_non_fire} (Total: {test_fire + test_non_fire})")

    # 2. Dynamic Class Weights (Calculated from Training set ONLY)
    # 0 = Fire, 1 = Non-Fire
    total_train = train_fire + train_non_fire
    weight_for_0 = total_train / (2.0 * train_fire)
    weight_for_1 = total_train / (2.0 * train_non_fire)
    class_weight = {0: weight_for_0, 1: weight_for_1}

    print("\nCalculated Class Weights (Training set only):")
    print(f"  Class 0 (Fire):     {weight_for_0:.4f}")
    print(f"  Class 1 (Non-Fire): {weight_for_1:.4f}")

    # 3. Data Pipeline Parameters
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    CLASS_NAMES = ["Fire", "Non-Fire"]

    print("\nLoading datasets with tf.keras.utils.image_dataset_from_directory...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='binary',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='binary',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False
    )

    # Pipeline optimization
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    # 4. Custom CNN Architecture
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Rescaling(1./255),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name="Custom_CNN_Forest_Guardian")

    # 5. Compile Model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    model.summary()

    # 6. Callbacks
    model_checkpoint_file = "custom_cnn_model.h5"
    history_csv_file = "custom_cnn_history.csv"

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_checkpoint_file,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1
        ),
        callbacks.CSVLogger(
            filename=history_csv_file,
            separator=',',
            append=False
        )
    ]

    # 7. Training
    print("\n" + "=" * 60)
    print("STARTING TRAINING (Max 15 Epochs)...")
    print("=" * 60)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        class_weight=class_weight,
        callbacks=cb_list,
        verbose=1
    )

    # Load best model for test evaluation
    if os.path.exists(model_checkpoint_file):
        print(f"\nLoading best saved model from '{model_checkpoint_file}' for evaluation...")
        model = models.load_model(model_checkpoint_file)

    # 8. Evaluation on untouched Test dataset
    print("\n" + "=" * 60)
    print("EVALUATING ON UNTOUCHED TEST DATASET")
    print("=" * 60)

    # Gather ground truth labels
    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0).flatten().astype(int)
    
    # Predict probabilities P(Non-Fire)
    y_pred_prob = model.predict(test_ds, verbose=1).flatten()

    # Apply threshold: >= 0.5 -> 1 (Non-Fire), < 0.5 -> 0 (Fire)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Calculate Metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average='binary')) # for label 1 (Non-Fire)
    rec = float(recall_score(y_true, y_pred, average='binary'))
    f1 = float(f1_score(y_true, y_pred, average='binary'))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro'))
    roc_auc = float(roc_auc_score(y_true, y_pred_prob))

    # Confusion matrix with explicit labels order: [0 (Fire), 1 (Non-Fire)]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_list = cm.tolist()

    # Class 0 (Fire) Specific Metrics
    fire_prec = float(precision_score(y_true, y_pred, pos_label=0))
    fire_rec = float(recall_score(y_true, y_pred, pos_label=0))
    fire_f1 = float(f1_score(y_true, y_pred, pos_label=0))
    fire_fn = int(cm[0][1]) # True Fire predicted as Non-Fire

    # Print Final Test Metrics
    print("\n--- FINAL TEST METRICS ---")
    print(f"  Overall Accuracy:      {acc:.4f}")
    print(f"  Precision (Non-Fire):  {prec:.4f}")
    print(f"  Recall (Non-Fire):     {rec:.4f}")
    print(f"  F1-Score (Non-Fire):   {f1:.4f}")
    print(f"  Macro F1-Score:        {macro_f1:.4f}")
    print(f"  ROC-AUC Score:         {roc_auc:.4f}")
    print(f"\n--- FIRE CLASS (LABEL 0) METRICS ---")
    print(f"  Fire Precision:        {fire_prec:.4f}")
    print(f"  Fire Recall:           {fire_rec:.4f}")
    print(f"  Fire F1-Score:         {fire_f1:.4f}")
    print(f"  Fire False Negatives:  {fire_fn}")
    print("\n--- CONFUSION MATRIX (Rows: True, Cols: Predicted) ---")
    print("                 Pred Fire (0)   Pred Non-Fire (1)")
    print(f"  True Fire (0):      {cm[0][0]:<15} {cm[0][1]:<15}")
    print(f"  True Non-Fire (1):  {cm[1][0]:<15} {cm[1][1]:<15}")

    # 9. Save Metrics to JSON
    metrics_json_file = "custom_cnn_test_metrics.json"
    metrics_data = {
        "class_mapping": {
            "0": "Fire",
            "1": "Non-Fire"
        },
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm_list,
        "fire_precision": fire_prec,
        "fire_recall": fire_rec,
        "fire_f1": fire_f1,
        "fire_false_negatives": fire_fn
    }

    with open(metrics_json_file, 'w') as f:
        json.dump(metrics_data, f, indent=4)

    print("\n" + "=" * 60)
    print("SAVED ARTIFACTS:")
    print(f"  1. Model Weights:     {model_checkpoint_file}")
    print(f"  2. Training History:  {history_csv_file}")
    print(f"  3. Test Metrics JSON: {metrics_json_file}")
    print("=" * 60)

if __name__ == "__main__":
    train_custom_cnn()
