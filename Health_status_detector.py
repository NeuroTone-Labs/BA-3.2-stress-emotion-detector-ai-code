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

