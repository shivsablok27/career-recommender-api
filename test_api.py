import requests
import json

# API Endpoint
url = "http://localhost:8000/predict"

# Data Science Profile (High I, O, Pattern Focus + Boosted Project/Team)
payload = {
  "personality_scores": {
    "R": 0.60, "I": 0.95, "A": 0.40, "S": 0.30, "E": 0.40, "C": 0.80,
    "O": 0.90, "C2": 0.85, "E2": 0.40, "A2": 0.50, "N": 0.30
  },
  "reading_responses": {
    "Qpattern1": 5,
    "Qpattern2": 5,
    "Qprobsolve1": 3,
    "Qprobsolve2": 3,
    "Qmgmt1": 2,
    "Qmgmt2": 2,
    "chosenActivity": "Discovering hidden trends or logical patterns",
    "freeText": "I love finding patterns in data...",
    "chosenProject": "Analyzing data to understand behaviour or predict outcomes",
    "team_choice": "Observation and analysis team"
  }
}

try:
    print("Sending request to API...")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("\nSUCCESS!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\nFAILED: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"\nERROR: {e}")
