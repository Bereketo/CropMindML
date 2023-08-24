import pandas as pd
from .model_training import preprocess_data, load_data
import joblib
import sys
import os
import django

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)
sys.path.append(project_root)


sys.path.append('/home/kali/dwr')
crop_app_path = os.path.join(project_root, 'backend', 'crop')
sys.path.append(crop_app_path)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.crop_mind.settings")
django.setup()


def load_model(filename):
    """Load a pre-trained model from disk."""
    return joblib.load(filename)


def predict_samples(model, sample_data, label_encoder):
    """Make predictions using the pre-trained model."""
    y_pred_onehot = model.predict(sample_data)
    y_pred_encoded = y_pred_onehot.argmax(axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    return y_pred

