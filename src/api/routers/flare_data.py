"""
Router for solar flare data endpoints.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.gcp_storage import GCPStorageClient
from utils.NASA_API import NASADonkiAPI

# Models
class FlareDetail(BaseModel):
    flare_id: str
    start_time: str
    peak_time: Optional[str] = None
    end_time: Optional[str] = None
    class_type: str
    source_location: Optional[str] = None
    active_region: Optional[str] = None
    intensity: Optional[float] = None
    link: Optional[str] = None

class FlareStats(BaseModel):
    total_count: int
    class_distribution: Dict[str, int]
    time_range: Dict[str, str]
    strongest_flare: Optional[FlareDetail] = None

# Router
router = APIRouter(
    prefix="/flares",
    tags=["flares"],
    responses={404: {"description": "Not found"}},
)

# Dependencies
def get_nasa_api():
    api_key = os.getenv("NASA_API_KEY", "DEMO_KEY")
    return NASADonkiAPI(api_key=api_key)

def get_gcs_client():
    project_id = os.getenv("GCP_PROJECT_ID")
    if project_id:
        return GCPStorageClient(project_id=project_id)
    return None

@router.get("/recent", response_model=List[FlareDetail])
async def get_recent_flares(
    days: int = Query(7, description="Number of days to look back"),
    class_filter: Optional[str] = Query(None, description="Filter by flare class (X, M, C, B)"),
    nasa_api: NASADonkiAPI = Depends(get_nasa_api)
):
    """Get recent solar flares from NASA DONKI API."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for API
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Get flare data from NASA API
        flares = nasa_api.get_flr_data(start_date=start_str, end_date=end_str)
        
        # Transform to response format
        result = []
        for flare in flares:
            flare_class = flare.get("classType", "")
            
            # Apply class filter if specified
            if class_filter and not flare_class.startswith(class_filter):
                continue
                
            result.append(FlareDetail(
                flare_id=flare.get("flrID", "unknown"),
                start_time=flare.get("beginTime", ""),
                peak_time=flare.get("peakTime"),
                end_time=flare.get("endTime"),
                class_type=flare_class,
                source_location=flare.get("sourceLocation"),
                active_region=flare.get("activeRegionNum"),
                intensity=_extract_intensity(flare_class),
                link=flare.get("link")
            ))
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching flare data: {str(e)}")

@router.get("/stats", response_model=FlareStats)
async def get_flare_statistics(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    gcs_client: Optional[GCPStorageClient] = Depends(get_gcs_client),
    nasa_api: NASADonkiAPI = Depends(get_nasa_api)
):
    """Get statistics about solar flares in a given time period."""
    try:
        # Try to get data from GCS first
        flares = []
        bucket_name = os.getenv("GCS_BUCKET_NAME", "solarguardai-data")
        
        if gcs_client:
            try:
                blob_path = f"data/raw/flares_{start_date}_{end_date}.json"
                data = gcs_client.download_json(bucket_name, blob_path)
                flares = data.get("flares", [])
            except Exception:
                # If not found in GCS, fetch from NASA API
                flares = nasa_api.get_flr_data(start_date=start_date, end_date=end_date)
        else:
            # Fetch from NASA API
            flares = nasa_api.get_flr_data(start_date=start_date, end_date=end_date)
        
        # Calculate statistics
        total_count = len(flares)
        
        # Class distribution
        class_distribution = {"X": 0, "M": 0, "C": 0, "B": 0, "A": 0, "Other": 0}
        strongest_flare = None
        strongest_intensity = -1
        
        for flare in flares:
            flare_class = flare.get("classType", "")
            
            # Update class distribution
            if flare_class.startswith("X"):
                class_distribution["X"] += 1
            elif flare_class.startswith("M"):
                class_distribution["M"] += 1
            elif flare_class.startswith("C"):
                class_distribution["C"] += 1
            elif flare_class.startswith("B"):
                class_distribution["B"] += 1
            elif flare_class.startswith("A"):
                class_distribution["A"] += 1
            else:
                class_distribution["Other"] += 1
            
            # Check if this is the strongest flare
            intensity = _extract_intensity(flare_class)
            if intensity > strongest_intensity:
                strongest_intensity = intensity
                strongest_flare = FlareDetail(
                    flare_id=flare.get("flrID", "unknown"),
                    start_time=flare.get("beginTime", ""),
                    peak_time=flare.get("peakTime"),
                    end_time=flare.get("endTime"),
                    class_type=flare_class,
                    source_location=flare.get("sourceLocation"),
                    active_region=flare.get("activeRegionNum"),
                    intensity=intensity,
                    link=flare.get("link")
                )
        
        return FlareStats(
            total_count=total_count,
            class_distribution=class_distribution,
            time_range={"start": start_date, "end": end_date},
            strongest_flare=strongest_flare
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating flare statistics: {str(e)}")

@router.get("/by-region/{region_id}", response_model=List[FlareDetail])
async def get_flares_by_region(
    region_id: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    nasa_api: NASADonkiAPI = Depends(get_nasa_api)
):
    """Get flares from a specific active region."""
    try:
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Get flare data from NASA API
        flares = nasa_api.get_flr_data(start_date=start_date, end_date=end_date)
        
        # Filter by region
        region_flares = []
        for flare in flares:
            if str(flare.get("activeRegionNum", "")) == region_id:
                region_flares.append(FlareDetail(
                    flare_id=flare.get("flrID", "unknown"),
                    start_time=flare.get("beginTime", ""),
                    peak_time=flare.get("peakTime"),
                    end_time=flare.get("endTime"),
                    class_type=flare.get("classType", ""),
                    source_location=flare.get("sourceLocation"),
                    active_region=flare.get("activeRegionNum"),
                    intensity=_extract_intensity(flare.get("classType", "")),
                    link=flare.get("link")
                ))
        
        return region_flares
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching flares for region {region_id}: {str(e)}")

def _extract_intensity(class_type: str) -> float:
    """Extract numerical intensity from flare class."""
    try:
        if not class_type or len(class_type) < 2:
            return 0.0
        
        # Format is like "X1.2", "M5.5", etc.
        class_letter = class_type[0]
        magnitude = float(class_type[1:])
        
        # Convert to unified scale
        if class_letter == "X":
            return 10.0 + magnitude
        elif class_letter == "M":
            return 1.0 + (magnitude / 10.0)
        elif class_letter == "C":
            return 0.1 + (magnitude / 100.0)
        elif class_letter == "B":
            return 0.01 + (magnitude / 1000.0)
        elif class_letter == "A":
            return 0.001 + (magnitude / 10000.0)
        else:
            return 0.0
    except:
        return 0.0