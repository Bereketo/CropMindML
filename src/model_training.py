import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib  # Used for saving and loading models

def load_data(filename):
    """Load data from CSV file."""
    return pd.read_csv(filename)

def preprocess_data(data):
    """Preprocess data by separating features and target variables, and encoding labels."""
    X = data.drop(columns=['label'])
    y = data['label']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    onehot_encoder = OneHotEncoder(sparse=False)
    y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))

    return X, y_onehot, label_encoder

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    """Save the trained model to disk."""
    joblib.dump(model, filename)

def load_model(filename):
    """Load a pre-trained model from disk."""
    return joblib.load(filename)

def predict_samples(model, sample_data, label_encoder):
    """Make predictions using the pre-trained model."""
    y_pred_onehot = model.predict(sample_data)
    y_pred_encoded = y_pred_onehot.argmax(axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    return y_pred

def evaluate_model(model, X_test, y_test):
    """Evaluate model accuracy."""
    y_pred_encoded = model.predict(X_test)
    accuracy = accuracy_score(y_test.argmax(axis=1), y_pred_encoded.argmax(axis=1))
    return accuracy

def main():
    data = load_data('Crop_recommendation.csv')

    X, y_onehot, label_encoder = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    rf_model = train_model(X_train, y_train)

    save_model(rf_model, '../models/crop_rf_model.pkl')

    loaded_model = load_model('../models/crop_rf_model.pkl')

    sample_data = [[82, 36, 41, 23.325013100000003, 79.79609448, 6.581693772, 187.3096148]]
    y_pred = predict_samples(loaded_model, sample_data, label_encoder)

    print(f'Predicted Crop: {y_pred[0]}')

if __name__ == "__main__":
    main()

