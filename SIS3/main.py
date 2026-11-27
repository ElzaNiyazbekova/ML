# main.py
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from models import penguin_features, PredictionOut 
from pathlib import Path

app = FastAPI(
    title="Penguin Prediction Service",
    description="A simple API to predict Penguin species using a pre-trained model.",
    version="0.1.0",
)

model_path = Path(__file__).parent / "log_reg.joblib"
try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
    class_names = ['Adelie', 'Chinstrap', 'Gentoo']
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    model = None 
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/")
def read_root():
    """Root endpoint providing basic API information."""
    return {"message": "Welcome to the Penguin Prediction API!"}

@app.post("/predict", response_model=PredictionOut)
async def predict_penguin(features: penguin_features):
    """
    Predict the Penguin species based on input features.

    Takes sepal length, sepal width, petal length, and petal width,
    returns the predicted class ID, class name, and probabilities.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or unavailable.")
    input_data = np.array([[
        features.bill_length_mm,
        features.bill_depth_mm,
        features.flipper_length_mm,
        features.body_mass_g
    ]])

    try:
        prediction = str(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return PredictionOut(
        predicted_class_name=prediction,
        probabilities=probabilities
    )
    

# --- Run the Application (Optional, for direct execution) ---
# Typically, you'd run this using Uvicorn from the command line.
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

