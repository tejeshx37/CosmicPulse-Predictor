"""
Quick test script for SolarGuardAI core functionality
"""
import requests
from datetime import datetime, timedelta

def test_nasa_api():
    """Test the NASA DONKI API for solar flare data"""
    print("Testing NASA DONKI API...")
    
    # Set date range for last 30 days
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    # NASA DONKI API endpoint for solar flares
    url = f"https://api.nasa.gov/DONKI/FLR?startDate={start_date}&endDate={end_date}&api_key=DEMO_KEY"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Successfully connected to NASA API")
            print(f"✅ Retrieved {len(data)} solar flare records")
            
            if len(data) > 0:
                print("\nSample flare data:")
                flare = data[0]
                print(f"  Flare ID: {flare.get('flrID')}")
                print(f"  Class: {flare.get('classType')}")
                print(f"  Begin Time: {flare.get('beginTime')}")
                print(f"  Active Region: {flare.get('activeRegionNum')}")
            
            return True
        else:
            print(f"❌ API request failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== SolarGuardAI Quick Test ===")
    success = test_nasa_api()
    
    if success:
        print("\n✅ Core functionality is working!")
        print("The NASA API connection is successful, which is the foundation of the project.")
    else:
        print("\n❌ Test failed. Please check the error messages above.")