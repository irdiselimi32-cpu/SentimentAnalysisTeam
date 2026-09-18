import os
import eel
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "base_model", "logistic_regression.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "base_model", "tfidf_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

label_map = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}

@eel.expose
def predict_sentiment(comment):
    if not comment or comment.strip() == "":
        return {
            "label": "Empty",
            "confidence": 0,
            "message": "Please enter a comment."
        }

    X = vectorizer.transform([comment])
    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]
    confidence = float(np.max(probabilities) * 100)

    return {
        "label": label_map.get(int(prediction), "Unknown"),
        "confidence": round(confidence, 2),
        "message": f"Prediction: {label_map.get(int(prediction), 'Unknown')}"
    }

eel.init(os.path.dirname(os.path.abspath(__file__)))

eel.start(
    "index.html",
    size=(700, 600)
)