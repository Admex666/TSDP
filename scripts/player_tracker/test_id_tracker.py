"""
Test ID-based tracker API
"""
import requests
import json

# Test add players by ID
url = "http://localhost:8003/api/v1/tracker/add-players-by-id"
data = {
    "players": [
        {
            "name": "Test Player",
            "fbref_id": "test123",
            "sofascore_id": None
        }
    ]
}

print("Testing add-players-by-id endpoint...")
print(f"URL: {url}")
print(f"Data: {json.dumps(data, indent=2)}")

try:
    response = requests.post(url, json=data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"\nError: {e}")
    print(f"Response text: {response.text if 'response' in locals() else 'No response'}")
