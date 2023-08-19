import pandas as pd
from model_training import train_model
from model_training import preprocess_data, load_data
import joblib


def user_input():
    """gets the input of users """
    sample_data = []
    nitro = float(input("Enter nitro level: "))
    phos = float(input("Enter phos level: "))
    pota = float(input("Enter potas level: "))
    temp = float(input("Enter tempreature: "))
    humidity = float(input("Enter humidity level: "))
    ph       = float(input("Enter ph level: "))
    rainfall = float(input("Enter rainfall level: "))

    sample_data.append([nitro, phos, pota, temp, humidity, ph, rainfall])
    return sample_data

def load_model(filename):
    """Load a pre-trained model from disk."""
    return joblib.load(filename)


def predict_samples(model, sample_data, label_encoder):
    """Make predictions using the pre-trained model."""
    y_pred_onehot = model.predict(sample_data)
    y_pred_encoded = y_pred_onehot.argmax(axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    return y_pred

model = load_model('../models/crop_rf_model.pkl')
sample_data = [[23.325013100000003, 79.79609448, 6.581693772, 187.3096148]]
filename = load_data('../data/crop_data.csv')
X, y_onehot, label_encoder = preprocess_data(filename)

ans = predict_samples(model, sample_data, label_encoder)
print(ans)
