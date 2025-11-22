import requests
import json
import argparse

SERVER_URL = 'http://localhost:8000'

def main():
    parser = argparse.ArgumentParser(description='Send a query to the Server')
    parser.add_argument('--query', type=str, required=True, help='The query to search for')

    args = parser.parse_args()
    
    payload = {'query': args.query}

    try:
        response = requests.post(f"{SERVER_URL}/predict", json=payload)

        result = response.json()['output']['raw']

        print(json.dumps(result, indent=2))

    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")

if __name__ == "__main__":
    main()