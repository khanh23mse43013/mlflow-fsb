
from fastapi import APIRouter
from pydantic import BaseModel 
from src.services.predict_service import get_predict_data
import joblib

# Load the model
model = joblib.load("best_model.pkl")

# Initialize FastAPI app
predict_controller = APIRouter(prefix="/api/predict-best-model", tags=["Predict model"])

@predict_controller.post("/predict")
def predict():
    prediction = get_predict_data()
    return prediction
