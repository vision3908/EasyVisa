from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model
model = joblib.load('models/visa_model.pkl')
feature_names = joblib.load("models/feature_names.pkl")
# Create API
app = FastAPI(
    title="Visa Approval Prediction API",
    description="Predict visa approval probability",
    version="1.0.0"
)

# Define input data structure
class VisaApplication(BaseModel):
    continent: str
    education_of_employee: str
    has_job_experience: str
    requires_job_training: str
    no_of_employees: int
    yr_of_estab: int
    region_of_employment: str
    prevailing_wage: float
    unit_of_wage: str
    full_time_position: str
    
    # Example for API docs
    class Config:
        schema_extra = {
            "example": {
                "continent": "Asia",
                "education_of_employee": "Master's",
                "has_job_experience": "Y",
                "requires_job_training": "N",
                "no_of_employees": 500,
                "yr_of_estab": 2010,
                "region_of_employment": "West",
                "prevailing_wage": 85000.0,
                "unit_of_wage": "Yearly",
                "full_time_position": "Y"
            }
        }

# Health check endpoint
@app.get("/")
def home():
    return {
        "message": "Visa Approval Prediction API",
        "status": "running",
        "version": "1.0.0"
    }

# Prediction endpoint
@app.post("/predict")
def predict_visa(application: VisaApplication):
    try:
        # Convert input to dataframe
        input_data = pd.DataFrame([application.dict()])
        
        # Apply same preprocessing as training
        # (You'll need to add your preprocessing steps here)
        # For now, simplified version:
        input_data_encoded = pd.get_dummies(input_data, drop_first=True)
        
        # Ensure columns match training data
        for col in feature_names:
            if col not in input_data_encoded.columns:
                input_data_encoded[col] = 0
        input_data_encoded = input_data_encoded[feature_names]
        
        # Make prediction
        prediction = model.predict(input_data_encoded)[0]
        probability = model.predict_proba(input_data_encoded)[0]
        
        # Return result
        return {
            "prediction": "Certified" if prediction == 1 else "Denied",
            "probability_certified": float(probability[1]),
            "probability_denied": float(probability[0])
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "message": "Prediction failed"
        }

# Model info endpoint
@app.get("/model-info")
def model_info():
    return {
        "model_type": "Random Forest Classifier",
        "features": feature_names,
        "n_features": len(feature_names)
    }
