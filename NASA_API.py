import requests

API_KEY = 'YOUR_NASA_API_KEY'
url = 'https://api.nasa.gov/DONKI/FLR'
params = {
    'startDate': '2022-01-01',
    'endDate': '2022-01-31',
    'api_key': API_KEY
}

response = requests.get(url, params=params)
data = response.json()

for event in data:
    print(f"Flare ID: {event['flrID']}")
    print(f"Start Time: {event['beginTime']}, Peak Time: {event['peakTime']}, Class: {event['classType']}")
    print('-'*50)
