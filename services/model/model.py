from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

MODEL_PATH = "model.joblib"

def train_model():
    print("Loading dataset...")
    data = load_iris()
    X, y = data.data, data.target

    print("Training model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)

    print("Training complete.")

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Training new model...")
        train_model()

    print("Loading model...")
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    train_model()
