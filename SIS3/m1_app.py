# models.py
from pydantic import BaseModel, Field
from typing import List

class IrisFeatures(BaseModel):
    """Input features for Iris prediction."""
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")

    class Config:
        # Example for FastAPI documentation
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class PredictionOut(BaseModel):
    """Prediction output schema."""
    predicted_class_id: int = Field(..., description="Predicted class index (0, 1, or 2)")
    predicted_class_name: str = Field(..., description="Predicted class name ('setosa', 'versicolor', 'virginica')")
    probabilities: List[float] = Field(..., description="List of probabilities for each class [setosa, versicolor, virginica]")

