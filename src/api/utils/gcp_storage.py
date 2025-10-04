"""
GCP Storage Client for SolarGuardAI
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

class GCPStorageClient:
    """
    A simplified GCP Storage Client for the Docker container
    """
    def __init__(self, project_id: str = None, bucket_name: str = None):
        """Initialize the GCP Storage Client"""
        self.project_id = project_id or os.environ.get('GCP_PROJECT_ID', 'solarguardai-project')
        self.bucket_name = bucket_name or os.environ.get('GCS_BUCKET_NAME', 'solarguardai-data')
        
    def upload_blob(self, source_data: Union[str, dict, list], destination_blob_name: str) -> str:
        """Mock upload to GCS bucket"""
        print(f"Mock: Uploaded data to {destination_blob_name}")
        return destination_blob_name
        
    def download_blob(self, source_blob_name: str) -> Optional[Union[Dict, List]]:
        """Mock download from GCS bucket"""
        print(f"Mock: Downloaded data from {source_blob_name}")
        
        # Return mock data based on the requested blob name
        if 'flare_data' in source_blob_name:
            return [
                {
                    "flrID": "2025-09-04T05:22:00-FLR-001",
                    "classType": "C9.2",
                    "beginTime": "2025-09-04T05:22Z",
                    "peakTime": "2025-09-04T05:40Z",
                    "endTime": "2025-09-04T06:15Z",
                    "sourceLocation": "N12E22",
                    "activeRegionNum": 14207
                },
                {
                    "flrID": "2025-09-03T12:15:00-FLR-001",
                    "classType": "M2.3",
                    "beginTime": "2025-09-03T12:15Z",
                    "peakTime": "2025-09-03T12:30Z",
                    "endTime": "2025-09-03T13:05Z",
                    "sourceLocation": "S08W14",
                    "activeRegionNum": 14205
                }
            ]
        elif 'model' in source_blob_name:
            return {
                "model_type": "xgboost",
                "version": "1.0.0",
                "accuracy": 0.85,
                "f1_score": 0.82,
                "training_date": datetime.now().isoformat()
            }
        else:
            return None
            
    def list_blobs(self, prefix: str = None) -> List[str]:
        """Mock list blobs in GCS bucket"""
        if prefix == 'models/':
            return ['models/xgboost_v1.pkl', 'models/random_forest_v1.pkl']
        elif prefix == 'data/':
            return ['data/flare_data_2025-09.json', 'data/flare_data_2025-08.json']
        else:
            return []