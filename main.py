from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all ML models
MODEL_PATHS = {
    "rf": "models/crop_model_rf.joblib",
    "svm": "models/crop_model_svm.joblib",
    "lr": "models/crop_model_lr.joblib",
    "knn": "models/crop_model_knn.joblib"
}

# Load the scaler
SCALER_PATH = "models/scaler.joblib"
if not os.path.exists(SCALER_PATH):
    raise RuntimeError("Scaler not found! Please run train_models.py first.")
scaler = joblib.load(SCALER_PATH)

# Load feature names
FEATURE_NAMES_PATH = "models/feature_names.joblib"
if not os.path.exists(FEATURE_NAMES_PATH):
    raise RuntimeError("Feature names not found! Please run train_models.py first.")
feature_names = joblib.load(FEATURE_NAMES_PATH)

# Load models
models = {}
for key, path in MODEL_PATHS.items():
    if os.path.exists(path):
        models[key] = joblib.load(path)
    else:
        print(f"Warning: Model file for {key} not found at {path}")

# Ensure at least the default model (Random Forest) is available
if "rf" not in models:
    raise RuntimeError("Default Random Forest model not found! Please run train_models.py first.")

class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Crop details dictionary
CROP_DETAILS = {
    "rice": {
        "growing_season": "120-140 days",
        "water_needs": "High (150-300 mm/month)",
        "soil_type": "Clay loam with good water retention",
        "fertilizer": f"N:P:K - High nitrogen (N), moderate phosphorus (P) and potassium (K)",
        "optimal_conditions": "pH: 6.0-6.8, Temp: 24-28°C, Humidity: 75-85%",
        "nutrient_needs": "N: 80-100 kg/ha, P: 30-50 kg/ha, K: 40-60 kg/ha"
    },
    "wheat": {
        "growing_season": "100-150 days",
        "water_needs": "Moderate (450-650 mm total)",
        "soil_type": "Well-drained loamy soil",
        "fertilizer": f"N:P:K - Balanced ratio with moderate nitrogen",
        "optimal_conditions": "pH: 5.8-6.2, Temp: 18-22°C, Humidity: 60-65%",
        "nutrient_needs": "N: 100-120 kg/ha, P: 50-60 kg/ha, K: 40-50 kg/ha"
    },
    "maize": {
        "growing_season": "90-120 days",
        "water_needs": "Moderate to High (500-800 mm total)",
        "soil_type": "Well-drained loam or sandy loam",
        "fertilizer": f"N:P:K - High nitrogen requirement",
        "optimal_conditions": "pH: 5.8-6.2, Temp: 22-25°C, Humidity: 65-75%",
        "nutrient_needs": "N: 135-200 kg/ha, P: 62-75 kg/ha, K: 45-100 kg/ha"
    },
    "chickpea": {
        "growing_season": "90-120 days",
        "water_needs": "Low to Moderate (400-600 mm total)",
        "soil_type": "Well-drained loamy soil",
        "fertilizer": "N:P:K - Low nitrogen (fixes own N), needs P",
        "optimal_conditions": "pH: 6.0-8.0, Temp: 20-25°C, Humidity: 60-70%",
        "nutrient_needs": "N: 20-30 kg/ha, P: 40-60 kg/ha, K: 20-30 kg/ha"
    },
    "kidneybeans": {
        "growing_season": "85-120 days",
        "water_needs": "Moderate (350-500 mm total)",
        "soil_type": "Well-drained, rich loamy soil",
        "fertilizer": "N:P:K - Low nitrogen, moderate P and K",
        "optimal_conditions": "pH: 6.0-6.5, Temp: 20-25°C, Humidity: 60-65%",
        "nutrient_needs": "N: 30-50 kg/ha, P: 60-75 kg/ha, K: 40-60 kg/ha"
    }
}

@app.get("/")
def read_root():
    return {
        "message": "Crop Recommendation API is running",
        "available_models": list(models.keys())
    }

@app.get("/models")
def get_available_models():
    return {
        "available_models": list(models.keys()),
        "model_descriptions": {
            "rf": "Random Forest - Good all-around model with feature importance",
            "svm": "Support Vector Machine - Good for complex decision boundaries",
            "lr": "Logistic Regression - Simple and interpretable",
            "knn": "K-Nearest Neighbors - Good for pattern-based prediction"
        }
    }

@app.post("/predict")
def predict_crop(data: CropInput, model_name: str = Query("rf", enum=list(models.keys()))):
    try:
        if model_name not in models:
            raise HTTPException(status_code=400, detail=f"Model {model_name} not available")

        # Prepare input data
        input_data = np.array([[
            data.N,
            data.P,
            data.K,
            data.temperature,
            data.humidity,
            data.ph,
            data.rainfall
        ]])

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Get the selected model
        model = models[model_name]

        # Make prediction
        predicted_crop = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Get feature importance (only available for Random Forest)
        feature_importance = {}
        if model_name == "rf" and hasattr(model, "feature_importances_"):
            feature_importance = dict(zip(feature_names, model.feature_importances_))

        # Get crop details with default values if not found
        default_crop_info = {
            "growing_season": "Varies by variety and region",
            "water_needs": "Moderate",
            "soil_type": "Well-drained soil",
            "fertilizer": "Balanced N:P:K ratio",
            "optimal_conditions": f"pH: {data.ph}, Temp: {data.temperature}°C, Humidity: {data.humidity}%",
            "nutrient_needs": f"N: {data.N} kg/ha, P: {data.P} kg/ha, K: {data.K} kg/ha"
        }
        
        crop_info = CROP_DETAILS.get(predicted_crop.lower(), default_crop_info)
        crop_optimal = crop_info["optimal_conditions"]
        nutrient_needs = crop_info["nutrient_needs"]
        
        response = {
            "model_used": model_name.upper(),
            "predicted_crop": predicted_crop.title(),
            "probabilities": dict(zip(
                [c.title() for c in model.classes_], 
                [round(p * 100, 1) for p in probabilities]
            )),
            "crop_details": crop_info,
            "input_analysis": {
                "provided_conditions": {
                    "N": f"{data.N} kg/ha",
                    "P": f"{data.P} kg/ha",
                    "K": f"{data.K} kg/ha",
                    "pH": data.ph,
                    "Temperature": f"{data.temperature}°C",
                    "Humidity": f"{data.humidity}%",
                    "Rainfall": f"{data.rainfall}mm"
                },
                "optimal_conditions": crop_optimal,
                "nutrient_requirements": nutrient_needs
            }
        }

        # Add feature importance if available
        if feature_importance:
            response["feature_importance"] = feature_importance

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 