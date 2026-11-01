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
