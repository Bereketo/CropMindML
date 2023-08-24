import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib


def load_data(filename):
    """Load data from CSV file."""
    return pd.read_csv(filename)

def preprocess_data(data):
    """Preprocess data by separating features and target variables, and encoding labels."""
    X = data.drop(columns=['label'])
    y = data['label']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    onehot_encoder = OneHotEncoder(sparse_output=False)
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


def main():
    data = load_data('../data/crop_data.csv')

    X, y_onehot, label_encoder = preprocess_data(data)
    label_encoder = label_encoder
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    rf_model = train_model(X_train, y_train)

    save_model(rf_model, '../models/crop_rf_model.pkl')

if __name__ == "__main__":
    main()

