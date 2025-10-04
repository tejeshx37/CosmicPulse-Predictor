"""
FastAPI backend for SolarGuardAI model serving.
This module provides API endpoints for solar flare prediction and data retrieval.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gcp_storage import GCPStorageClient
from models.prediction import ModelPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SolarGuardAI API",
    description="API for solar flare prediction and data retrieval",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Features for prediction")
    model_version: Optional[str] = Field(None, description="Model version to use")

class PredictionResponse(BaseModel):
    prediction: Any = Field(..., description="Prediction result")
    probability: float = Field(..., description="Prediction probability")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")

class FlareData(BaseModel):
    flare_id: str
    start_time: str
    peak_time: Optional[str] = None
    end_time: Optional[str] = None
    class_type: str
    source_location: Optional[str] = None
    active_region: Optional[str] = None
    link: Optional[str] = None

class FlareDataResponse(BaseModel):
    flares: List[FlareData]
    count: int
    start_date: str
    end_date: str

# Global variables
MODEL_CACHE = {}
GCS_CLIENT = None
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "solarguardai-data")

# Dependency for GCS client
def get_gcs_client():
    global GCS_CLIENT
    if GCS_CLIENT is None and PROJECT_ID:
        GCS_CLIENT = GCPStorageClient(project_id=PROJECT_ID)
    return GCS_CLIENT

# Dependency for model predictor
def get_model_predictor(model_version: str = None):
    """Get or load model predictor based on version."""
    if model_version in MODEL_CACHE:
        return MODEL_CACHE[model_version]
    
    # Default to latest model if version not specified
    if not model_version:
        # Logic to find latest model version
        model_version = "latest"
    
    try:
        predictor = ModelPredictor(model_version=model_version)
        MODEL_CACHE[model_version] = predictor
        return predictor
    except Exception as e:
        logger.error(f"Error loading model {model_version}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "name": "SolarGuardAI API",
        "version": "1.0.0",
        "description": "API for solar flare prediction and data retrieval",
        "endpoints": [
            {"path": "/predict", "method": "POST", "description": "Predict solar flare occurrence"},
            {"path": "/flares", "method": "GET", "description": "Get historical solar flare data"},
            {"path": "/models", "method": "GET", "description": "List available prediction models"},
            {"path": "/health", "method": "GET", "description": "API health check"}
        ]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    predictor: ModelPredictor = Depends(get_model_predictor)
):
    """Predict solar flare occurrence based on input features."""
    try:
        # Convert features to format expected by model
        features_df = pd.DataFrame([request.features])
        
        # Make prediction
        prediction, probability = predictor.predict(features_df)
        
        return PredictionResponse(
            prediction=prediction,
            probability=float(probability),
            model_version=predictor.model_version,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/flares", response_model=FlareDataResponse)
def get_flares(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    class_type: Optional[str] = Query(None, description="Filter by flare class (e.g., 'X', 'M', 'C')"),
    limit: int = Query(100, description="Maximum number of records to return"),
    gcs_client: GCPStorageClient = Depends(get_gcs_client)
):
    """Get historical solar flare data within a date range."""
    try:
        # Validate dates
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            if start > end:
                raise ValueError("Start date must be before end date")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
        
        # Fetch data from GCS or local storage
        flares = []
        
        if gcs_client:
            # Logic to fetch from GCS
            blob_path = f"data/raw/flares_{start_date}_{end_date}.json"
            try:
                data = gcs_client.download_json(BUCKET_NAME, blob_path)
                flares = data.get("flares", [])
            except Exception:
                # If specific file not found, try to find individual day files
                current_date = start
                while current_date <= end:
                    date_str = current_date.strftime("%Y-%m-%d")
                    try:
                        daily_blob = f"data/raw/flares_{date_str}.json"
                        daily_data = gcs_client.download_json(BUCKET_NAME, daily_blob)
                        flares.extend(daily_data.get("flares", []))
                    except Exception:
                        pass  # Skip if file not found
                    current_date += timedelta(days=1)
        else:
            # Fallback to local data or mock data
            logger.warning("GCS client not available, using mock data")
            # Generate some mock data for testing
            flares = generate_mock_flare_data(start, end)
        
        # Apply filters
        if class_type:
            flares = [f for f in flares if f.get("class_type", "").startswith(class_type)]
        
        # Apply limit
        flares = flares[:limit]
        
        # Convert to response format
        response_flares = []
        for flare in flares:
            response_flares.append(FlareData(
                flare_id=flare.get("flare_id", "unknown"),
                start_time=flare.get("start_time", ""),
                peak_time=flare.get("peak_time"),
                end_time=flare.get("end_time"),
                class_type=flare.get("class_type", "unknown"),
                source_location=flare.get("source_location"),
                active_region=flare.get("active_region"),
                link=flare.get("link")
            ))
        
        return FlareDataResponse(
            flares=response_flares,
            count=len(response_flares),
            start_date=start_date,
            end_date=end_date
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving flare data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving flare data: {str(e)}")

@app.get("/models")
def list_models(gcs_client: GCPStorageClient = Depends(get_gcs_client)):
    """List available prediction models."""
    try:
        models = []
        
        if gcs_client:
            # List models from GCS
            blobs = gcs_client.list_blobs(BUCKET_NAME, prefix="models/")
            
            # Group by model version
            model_versions = {}
            for blob in blobs:
                # Extract version from path like "models/v1/model.pkl"
                parts = blob.name.split('/')
                if len(parts) >= 3:
                    version = parts[1]
                    if version not in model_versions:
                        model_versions[version] = {
                            "version": version,
                            "files": [],
                            "created_at": blob.time_created.isoformat() if hasattr(blob, 'time_created') else None
                        }
                    model_versions[version]["files"].append(blob.name)
            
            models = list(model_versions.values())
        else:
            # Mock data if GCS not available
            models = [
                {"version": "v1", "files": ["model.pkl", "metadata.json"], "created_at": datetime.now().isoformat()},
                {"version": "v2", "files": ["model.pkl", "metadata.json"], "created_at": datetime.now().isoformat()}
            ]
        
        return {"models": models, "count": len(models)}
    
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

def generate_mock_flare_data(start_date, end_date):
    """Generate mock flare data for testing."""
    flares = []
    current_date = start_date
    
    flare_classes = ["X", "M", "C", "B"]
    class_weights = [0.05, 0.15, 0.3, 0.5]  # Probability distribution
    
    while current_date <= end_date:
        # Random number of flares per day (0-5)
        num_flares = np.random.randint(0, 6)
        
        for i in range(num_flares):
            # Random flare class based on weights
            flare_class = np.random.choice(flare_classes, p=class_weights)
            
            # Random magnitude within class
            if flare_class == "X":
                magnitude = round(np.random.uniform(1.0, 9.9), 1)
            else:
                magnitude = round(np.random.uniform(1.0, 9.9), 1)
            
            # Random times
            hour = np.random.randint(0, 24)
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            
            start_time = current_date.replace(hour=hour, minute=minute, second=second)
            
            # Duration between 10 minutes and 3 hours
            duration_minutes = np.random.randint(10, 180)
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            # Peak time between start and end
            peak_minutes = np.random.randint(1, duration_minutes)
            peak_time = start_time + timedelta(minutes=peak_minutes)
            
            # Active region (optional)
            if np.random.random() > 0.2:  # 80% chance to have active region
                active_region = f"AR{np.random.randint(11000, 13000)}"
            else:
                active_region = None
            
            flares.append({
                "flare_id": f"FL{current_date.strftime('%Y%m%d')}{i+1}",
                "start_time": start_time.isoformat(),
                "peak_time": peak_time.isoformat(),
                "end_time": end_time.isoformat(),
                "class_type": f"{flare_class}{magnitude}",
                "source_location": f"N{np.random.randint(0, 90)} W{np.random.randint(0, 90)}" if np.random.random() > 0.1 else None,
                "active_region": active_region,
                "link": f"https://www.solarmonitor.org/?date={current_date.strftime('%Y%m%d')}" if np.random.random() > 0.3 else None
            })
        
        current_date += timedelta(days=1)
    
    return flares

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)