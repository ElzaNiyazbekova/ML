# models.py
from pydantic import BaseModel, Field
from typing import List

class penguin_features(BaseModel):
    """Input features for Penguin prediction."""
    bill_length_mm: float = Field(..., gt=0, description="bill_length_mm")
    bill_depth_mm: float = Field(..., gt=0, description="bill_depth_mm")
    flipper_length_mm: float = Field(..., gt=0, description="flipper_length_mm")
    body_mass_g: float = Field(..., gt=0, description="body_mass_g")
#     species	island	bill_length_mm	bill_depth_mm	flipper_length_mm	body_mass_g	sex
# 0	Adelie	Torgersen	39.1	18.7	181.0	3750.0	Male
    class Config:
        # Example for FastAPI documentation
        schema_extra = {
            "example": {
                "bill_length_mm": 39.1,
                "bill_depth_mm": 18.7,
                "flipper_length_mm": 181.0,
                "body_mass_g": 3750.0
            }
        }
# 'Adelie', 'Chinstrap', 'Gentoo'
class PredictionOut(BaseModel):
    """Prediction output schema."""
    # predicted_class_id: int = Field(..., description="Predicted class index (0, 1, or 2)")
    predicted_class_name: str = Field(..., description="Predicted class name ('Adelie', 'Chinstrap', 'Gentoo')")
    probabilities: List[float] = Field(..., description="List of probabilities for each class [Adelie, Chinstrap, Gentoo]")

