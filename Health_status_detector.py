"""
Mental Health 1D CNN - Training and Prediction
==============================================

This script provides two modes for the 1D CNN mental health classification model:
- --train: trains the 1D CNN using the current training configuration
- --predict: loads the trained CNN and predicts a selected row from the CSV

Dataset: dataset/mental_health_multimodal.csv (4,000 samples)
Target: Mental_Health_Status (4 classes: Healthy, Mild_Stress, Moderate_Stress, Severe_Stress)
Features: 21 input features (3 scores + 18 behavioral/physiological)

IMPORTANT: This is an experimental ML evaluation, not a clinical assessment.
The dataset shows evidence that score features may be related to label construction.
Results should not be interpreted as evidence of real-world clinical validity.

Usage (Training):
    python predict_mental_health/mental_health_cnn.py --train

Usage (Prediction):
    python predict_mental_health/mental_health_cnn.py --predict --row 5
    python predict_mental_health/mental_health_cnn.py --predict --csv "dataset/mental_health_multimodal.csv" --row 5

NOTE: The --row argument refers to the original CSV row number (1-based), not the test set row number.
After training, you can check which split a row belongs to using the generated split index files:
- train_indices.csv: Original CSV row numbers in the training set
- val_indices.csv: Original CSV row numbers in the validation set
- test_indices.csv: Original CSV row numbers in the test set
"""

import html
import json
import re
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Configuration (relative to script location)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
CSV_PATH = PROJECT_DIR.parent / "dataset" / "mental_health_multimodal.csv"
OUTPUT_DIR = PROJECT_DIR / "mental_health_21_feature_cnn_experiment"
MODEL_PATH = OUTPUT_DIR / "best_model.keras"
PREPROCESSING_PATH = OUTPUT_DIR / "preprocessing.joblib"
TARGET_COLUMN = "Mental_Health_Status"
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20  # 20% of training data for validation
MAX_EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10

# Feature definitions
SCORE_FEATURES = [
    "Depression_Score",
    "Anxiety_Score",
    "Stress_Score",
]

BEHAVIORAL_FEATURES = [
    "Sleep_Quality",
    "Social_Engagement",
    "Daily_App_Usage_Min",
    "Typing_Speed_WPM",
    "Session_Frequency",
    "Idle_Time_Min",
    "Facial_Emotion_Variance",
    "Eye_Blink_Rate",
    "Smile_Intensity",
    "Head_Motion_Index",
    "MFCC_Mean",
    "MFCC_Variance",
    "Pitch_Mean",
    "Speech_Rate",
    "Heart_Rate_BPM",
    "HRV_Index",
    "Skin_Temperature",
    "GSR_Level",
]
FEATURE_COLUMNS = SCORE_FEATURES + BEHAVIORAL_FEATURES
EXPECTED_CLASSES = ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]

def clean_column_name(column_name):
    """Clean common formatting artifacts from CSV headers."""
    name = html.unescape(str(column_name)).strip()
    name = re.sub(r"^\*+|\*+$", "", name).strip()
    return name


def load_and_validate_data(csv_path):
    """Load dataset and validate required columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = pd.read_csv(csv_path)
    data.columns = [clean_column_name(column) for column in data.columns]

    # Validate target column
    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {', '.join(data.columns.tolist())}"
        )

    # Validate feature columns
    missing_features = [f for f in FEATURE_COLUMNS if f not in data.columns]
    if missing_features:
        raise ValueError(
            f"Missing required features: {', '.join(missing_features)}\n"
            f"Available columns: {', '.join(data.columns.tolist())}"
        )

    # Clean target labels
    data = data.dropna(subset=[TARGET_COLUMN]).copy()
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(str).str.strip()

    return data


def build_cnn_model(input_shape, num_classes):
    """Build a 1D CNN for tabular data."""
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),

        # First Conv1D block
        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            activation='relu',
            padding='same'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.3),

        # Second Conv1D block
        tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=3,
            activation='relu',
            padding='same'
        ),
        tf.keras.layers.BatchNormalization(),
        # Note: Skip pooling if dimension becomes too small
        tf.keras.layers.Dropout(0.4),

        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),

        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_mode():
    """Train the 1D CNN model."""
    print("=" * 76)
    print("NUEROTONE LABS: 1D CNN TRAINING FOR MENTAL HEALTH CLASSIFICATION")
    print("=" * 76)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CSV: {CSV_PATH}")
    print(f"Output: {OUTPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and validate data
    print("\nLoading and validating dataset...")
    data = load_and_validate_data(CSV_PATH)
    print(f"Total rows: {len(data)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    # Extract features and target
    X = data[FEATURE_COLUMNS].values
    y = data[TARGET_COLUMN].values
    original_indices = data.index.values  # Preserve original CSV row numbers (0-based)

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_.tolist()

    print(f"\nClass mapping:")
    for i, cls in enumerate(class_names):
        count = (y == cls).sum()
        pct = count / len(y) * 100
        print(f"  {i} -> {cls}: {count} ({pct:.1f}%)")

    # Verify target column is not in features
    if TARGET_COLUMN in FEATURE_COLUMNS:
        print(f"ERROR: Target column '{TARGET_COLUMN}' found in features!")
        return 1

    # Train/test split (80/20, stratified)
    print("\n" + "=" * 76)
    print("TRAIN/TEST SPLIT")
    print("=" * 76)

    X_train_full, X_test, y_train_full, y_test, idx_train_full, idx_test = train_test_split(
        X, y_encoded, original_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    print(f"Training+Validation: {len(X_train_full)} ({len(X_train_full)/len(X)*100:.1f}%)")
    print(f"Test (held-out): {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # Train/validation split (80/20 of training data, stratified)
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train_full, y_train_full, idx_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    print(f"\nTraining: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # Preprocessing (fit on training data only)
    print("\n" + "=" * 76)
    print("PREPROCESSING")
    print("=" * 76)

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    # Fit on training data
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    # Transform validation and test data
    X_val_imputed = imputer.transform(X_val)
    X_val_scaled = scaler.transform(X_val_imputed)

    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Reshape for CNN (samples, timesteps, features)
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    print(f"Input shape for CNN: {X_train_reshaped.shape}")
    print(f"Number of classes: {len(class_names)}")

    # Build model
    print("\n" + "=" * 76)
    print("BUILDING CNN MODEL")
    print("=" * 76)

    input_shape = (X_train_reshaped.shape[1], 1)
    num_classes = len(class_names)

    model = build_cnn_model(input_shape, num_classes)
    model.summary()

    # Callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    # Train model
    print("\n" + "=" * 76)
    print("TRAINING CNN")
    print("=" * 76)
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Training samples: {len(X_train_reshaped)}")
    print(f"Validation samples: {len(X_val_reshaped)}")

    start_time = time.time()

    history = model.fit(
        X_train_reshaped, y_train,
        validation_data=(X_val_reshaped, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint_callback, early_stopping],
        verbose=1
    )

    training_duration = time.time() - start_time
    print(f"\nTraining completed in {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")

    # Load best model
    print("\nLoading best model from checkpoint...")
    best_model = tf.keras.models.load_model(MODEL_PATH)

    # Evaluate on test set
    print("\n" + "=" * 76)
    print("TEST SET EVALUATION")
    print("=" * 76)

    test_loss, test_accuracy = best_model.evaluate(X_test_reshaped, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Get predictions
    y_pred_proba = best_model.predict(X_test_reshaped, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Macro F1:          {macro_f1:.4f}")
    print(f"  Weighted F1:       {weighted_f1:.4f}")
    print(f"  Balanced accuracy: {balanced_accuracy:.4f}")

    # Classification report
    report_dict = classification_report(
        y_test, y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test, y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0,
    )

    print(f"\nClassification Report:")
    print(report_text)

    # Confusion matrix
    matrix = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Actual_{name}" for name in class_names],
        columns=[f"Predicted_{name}" for name in class_names],
    )
    print(f"\nConfusion Matrix:")
    print(matrix_df.to_string())

    # Save preprocessing artifacts
    print("\n" + "=" * 76)
    print("SAVING ARTIFACTS")
    print("=" * 76)

    preprocessing_path = PREPROCESSING_PATH
    joblib.dump({
        'imputer': imputer,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_columns': FEATURE_COLUMNS,
        'target_column': TARGET_COLUMN,
        'train_indices': idx_train,
        'val_indices': idx_val,
        'test_indices': idx_test,
    }, preprocessing_path)
    print(f"Saved preprocessing: {preprocessing_path}")

    # Save split index files (original CSV row numbers)
    train_indices_path = OUTPUT_DIR / "train_indices.csv"
    val_indices_path = OUTPUT_DIR / "val_indices.csv"
    test_indices_path = OUTPUT_DIR / "test_indices.csv"

    pd.DataFrame({'csv_row_number': idx_train}).to_csv(train_indices_path, index=False)
    pd.DataFrame({'csv_row_number': idx_val}).to_csv(val_indices_path, index=False)
    pd.DataFrame({'csv_row_number': idx_test}).to_csv(test_indices_path, index=False)

    print(f"Saved train indices: {train_indices_path}")
    print(f"Saved validation indices: {val_indices_path}")
    print(f"Saved test indices: {test_indices_path}")

    # Save training history
    history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
    history_path = OUTPUT_DIR / "training_history.json"
    history_path.write_text(json.dumps(history_dict, indent=2), encoding="utf-8")
    print(f"Saved training history: {history_path}")

    # Save classification reports
    report_json_path = OUTPUT_DIR / "classification_report.json"
    report_json_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    report_txt_path = OUTPUT_DIR / "classification_report.txt"
    report_txt_path.write_text(report_text, encoding="utf-8")

    print(f"Saved classification report (JSON): {report_json_path}")
    print(f"Saved classification report (TXT): {report_txt_path}")

    # Save confusion matrix
    matrix_path = OUTPUT_DIR / "confusion_matrix.csv"
    matrix_df.to_csv(matrix_path)
    print(f"Saved confusion matrix: {matrix_path}")

    # Save test predictions with probabilities
    results_df = pd.DataFrame({
        'csv_row_number': idx_test,
        'actual_label': [class_names[y] for y in y_test],
        'predicted_label': [class_names[y] for y in y_pred],
        'correct': [y_test[i] == y_pred[i] for i in range(len(y_test))],
    })
    for i, cls in enumerate(class_names):
        results_df[f'prob_{cls}'] = y_pred_proba[:, i]

    results_path = OUTPUT_DIR / "test_predictions.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved test predictions: {results_path}")

    # Save metrics
    metrics = {
        "experiment_name": "mental_health_1d_cnn",
        "model_type": "1D Convolutional Neural Network",
        "feature_columns": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "class_names": class_names,
        "class_mapping": {str(i): cls for i, cls in enumerate(class_names)},
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
        "train_size": len(X_train),
        "validation_size_count": len(X_val),
        "test_size_count": len(X_test),
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "actual_epochs": len(history.history['loss']),
        "training_duration_seconds": training_duration,
        "input_shape": input_shape,
        "test_metrics": {
            "loss": float(test_loss),
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "balanced_accuracy": float(balanced_accuracy),
        },
        "class_metrics": {
            cls: {
                "precision": report_dict[cls]["precision"],
                "recall": report_dict[cls]["recall"],
                "f1_score": report_dict[cls]["f1-score"],
                "support": report_dict[cls]["support"],
            }
            for cls in class_names
        },
        "split_indices": {
            "train_indices_file": str(train_indices_path),
            "val_indices_file": str(val_indices_path),
            "test_indices_file": str(test_indices_path),
        },
        "warning": (
            "Experimental result only. Score features may be associated with "
            "target-label construction; results do not establish clinical validity. "
            "Dataset shows evidence of label leakage."
        ),
    }

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics: {metrics_path}")

    # Generate training plots
    matplotlib.use('Agg')  # Non-interactive backend

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "training_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training plots: {plot_path}")

    print("\n" + "=" * 76)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 76)
    return 0

def load_model_and_preprocessing():
    """Load the trained CNN model and preprocessing artifacts."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not PREPROCESSING_PATH.exists():
        raise FileNotFoundError(f"Preprocessing artifacts not found: {PREPROCESSING_PATH}")

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load preprocessing
    preprocessing = joblib.load(PREPROCESSING_PATH)

    return model, preprocessing


def load_csv_row(csv_path, row_number, preprocessing):
    """Load a specific row from the CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = [clean_column_name(col) for col in df.columns]

    # Validate row number (1-based)
    if row_number < 1:
        raise ValueError(
            f"Row number must be >= 1. Got: {row_number}"
        )

    if row_number > len(df):
        raise ValueError(
            f"Row number {row_number} is out of range. "
            f"CSV has {len(df)} data rows (valid range: 1-{len(df)})."
        )

    # Get feature columns from preprocessing
    feature_columns = preprocessing['feature_columns']
    target_column = preprocessing['target_column']

    # Check for required features
    missing_features = [f for f in feature_columns if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"CSV is missing required features: {', '.join(missing_features)}\n"
            f"Available columns: {', '.join(df.columns.tolist())}"
        )

    # Extract the row (0-based index)
    row_idx = row_number - 1
    row = df.iloc[row_idx]

    # Extract features in the correct order
    features_dict = {feat: row[feat] for feat in feature_columns}

    # Get actual label if available
    actual_label = None
    if target_column in df.columns:
        actual_label = str(row[target_column]).strip()

    return features_dict, actual_label, df


def predict_status(model, preprocessing, features_dict):
    """Make prediction using the loaded CNN model."""
    # Extract preprocessing components
    imputer = preprocessing['imputer']
    scaler = preprocessing['scaler']
    label_encoder = preprocessing['label_encoder']
    feature_columns = preprocessing['feature_columns']

    # Create feature vector in the correct order
    feature_vector = [features_dict[feat] for feat in feature_columns]
    X = np.array([feature_vector])

    # Apply preprocessing (same as training)
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    # Reshape for CNN (samples, timesteps, features)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Make prediction
    probabilities = model.predict(X_reshaped, verbose=0)[0]
    predicted_class_idx = np.argmax(probabilities)

    # Get class names from label encoder
    class_names = label_encoder.classes_.tolist()
    predicted_class = class_names[predicted_class_idx]

    # Create probability dictionary
    prob_dict = {
        class_name: float(prob)
        for class_name, prob in zip(class_names, probabilities)
    }

    result = {
        'predicted_class': predicted_class,
        'predicted_class_idx': int(predicted_class_idx),
        'probabilities': prob_dict,
        'class_names': class_names,
        'input_features': features_dict
    }

    return result


def display_results(result, actual_label=None, row_number=None, csv_path=None):
    """Display prediction results."""
    print("\n" + "=" * 76)
    print("NUEROTONE LABS - 1D CNN PREDICTION")
    print("=" * 76)

    if csv_path and row_number is not None:
        print(f"\nDataset: {csv_path}")
        print(f"Selected data row: {row_number}")

    print(f"\nPredicted category: {result['predicted_class']}")

    print(f"\nClass probabilities:")
    # Display in the order of class_names
    for class_name in result['class_names']:
        prob = result['probabilities'][class_name]
        print(f"{class_name:20s} {prob * 100:6.2f}%")

    if actual_label is not None:
        print(f"\nActual dataset label: {actual_label}")
        matches = result['predicted_class'] == actual_label
        print(f"Prediction matches dataset label: {'Yes' if matches else 'No'}")

    print("=" * 76)


def predict_mode(csv_path, row_number):
    """Run prediction mode."""
    print("Loading model and preprocessing...")
    model, preprocessing = load_model_and_preprocessing()
    print(f"Model loaded: {MODEL_PATH.name}")
    print(f"Preprocessing loaded: {PREPROCESSING_PATH.name}")
    print(f"Features: {len(preprocessing['feature_columns'])}")
    print(f"Classes: {', '.join(preprocessing['label_encoder'].classes_)}")

    # Load CSV row
    features_dict, actual_label, df = load_csv_row(csv_path, row_number, preprocessing)

    # Make prediction
    result = predict_status(model, preprocessing, features_dict)

    # Display results
    display_results(result, actual_label, row_number, csv_path)

    # Verify probabilities sum to ~1.0
    prob_sum = sum(result['probabilities'].values())
    if abs(prob_sum - 1.0) > 0.01:
        print(f"\nWarning: Probabilities sum to {prob_sum:.6f} (expected ~1.0)")

    return 0





