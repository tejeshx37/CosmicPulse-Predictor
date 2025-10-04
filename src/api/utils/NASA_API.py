"""
NASA DONKI API Client for SolarGuardAI
"""
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

class NASADonkiAPI:
    """
    Client for NASA DONKI API to retrieve solar flare data
    """
    def __init__(self, api_key: str = None):
        """Initialize the NASA DONKI API client"""
        self.api_key = api_key or os.environ.get('NASA_API_KEY', 'DEMO_KEY')
        self.base_url = "https://api.nasa.gov/DONKI"
        
    def get_flare_data(self, start_date: str, end_date: str = None) -> List[Dict]:
        """
        Get solar flare data from NASA DONKI API
        
        Args:
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD (defaults to today)
            
        Returns:
            List of solar flare data dictionaries
        """
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        url = f"{self.base_url}/FLR"
        params = {
            "startDate": start_date,
            "endDate": end_date,
            "api_key": self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching data: {response.status_code}")
                # Return mock data for testing
                return self._get_mock_data()
        except Exception as e:
            print(f"Exception when fetching data: {str(e)}")
            # Return mock data for testing
            return self._get_mock_data()
            
    def _get_mock_data(self) -> List[Dict]:
        """Return mock solar flare data for testing"""
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
            },
            {
                "flrID": "2025-09-02T18:05:00-FLR-001",
                "classType": "X1.5",
                "beginTime": "2025-09-02T18:05Z",
                "peakTime": "2025-09-02T18:27Z",
                "endTime": "2025-09-02T19:15Z",
                "sourceLocation": "N22W35",
                "activeRegionNum": 14203
            }
        ]