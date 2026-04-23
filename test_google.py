import requests
from utils.google_places import get_google_api_key

def test_google_api():
    api_key = get_google_api_key()
    print(f"Using API Key: {api_key[:10]}...{api_key[-5:]}")
    
    PLACES_SEARCH = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    query = "JOE'S PIZZA restaurant New York"
    
    print(f"Testing raw search query: {query}")
    try:
        response = requests.get(
            PLACES_SEARCH,
            params={
                "query": query,
                "key": api_key,
                "type": "restaurant",
                "region": "us",
            },
            timeout=10,
        )
        print(f"HTTP Status Code: {response.status_code}")
        
        data = response.json()
        status = data.get("status")
        print(f"Google API Status: {status}")
        
        if status != "OK":
            print(f"Error Message: {data.get('error_message', 'No error message provided by Google')}")
        else:
            results = data.get("results", [])
            print(f"Success! Found {len(results)} results.")
            if results:
                print(f"Top result name: {results[0].get('name')}")
                print(f"Top result Place ID: {results[0].get('place_id')}")
                
    except Exception as e:
        print(f"Request failed with exception: {e}")

if __name__ == "__main__":
    test_google_api()
