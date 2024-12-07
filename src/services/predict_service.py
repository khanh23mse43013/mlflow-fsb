import joblib
import numpy as np
from sklearn.datasets import make_regression
import pandas as pd

def get_predict_input():
    # Create a sample dataset
    X, y = make_regression(n_samples=1, n_features=20, noise=0.1, random_state=42)
    data = {"features": X[0].tolist()}
    return data

def load_model():
    model = joblib.load("best_model.pkl")
    return model

def get_predict_data():
    try:
        # Load the model
        model = load_model()
        # Get the input data
        data = get_predict_input()
        # Get the features
        features = np.array(data["features"]).reshape(1, -1)
        # Predict
        prediction = model.predict(features) # Convert to list for JSON compatibility
        print(f"Prediction: {prediction}")
    except Exception as e:
        print(f"Prediction: {prediction}")
        prediction = f"Error: {str(e)}"
    
    return f"prediction result: {prediction[0]}"
