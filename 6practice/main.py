# main.py
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from models import penguin_features, PredictionOut # Import Pydantic models
from pathlib import Path
model_path = Path(__file__).parent / "log_reg.joblib"
# --- Application Setup ---
app = FastAPI(
    title="Penguin Prediction Service",
    description="A simple API to predict Penguin species using a pre-trained model.",
    version="0.1.0",
)

# --- Model Loading ---
# Load the model at application startup.
# For larger applications, consider dependency injection.
model_path = "log_reg.joblib"
try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
    # Define class names based on the Iris dataset standard order
    class_names = ['Adelie', 'Chinstrap', 'Gentoo']
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    model = None # Set model to None if loading fails
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- API Endpoints ---
@app.get("/")
def read_root():
    """Root endpoint providing basic API information."""
    return {"message": "Welcome to the Penguin Prediction API!"}

@app.post("/predict", response_model=PredictionOut)
async def predict_iris(features: penguin_features):
    """
    Predict the Penguin species based on input features.

    Takes sepal length, sepal width, petal length, and petal width,
    returns the predicted class ID, class name, and probabilities.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or unavailable.")

    # 1. Convert input data to the format expected by the model
    #    (scikit-learn models usually expect a 2D NumPy array)
    input_data = np.array([[
        features.bill_length_mm,
        features.bill_depth_mm,
        features.flipper_length_mm,
        features.body_mass_g
    ]])

    # 2. Make prediction
    try:
        prediction = str(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return PredictionOut(
        predicted_class_name=prediction,
        probabilities=probabilities
    )
    # 3. Format the response
    # predicted_class_index = int(prediction_id[0]) # Get the first element

    # if predicted_class_index < 0 or predicted_class_index >= len(class_names):
    #      raise HTTPException(status_code=500, detail="Prediction index out of bounds.")

    predicted_class_name = class_names[predicted_class_index]
    prediction_probabilities = probabilities[0].tolist() # Get probabilities for the first (and only) input

    return PredictionOut(
        # predicted_class_id=predicted_class_index,
        predicted_class_name=predicted_class_name,
        probabilities=prediction_probabilities
    )

# --- Run the Application (Optional, for direct execution) ---
# Typically, you'd run this using Uvicorn from the command line.
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

