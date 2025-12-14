# Career Recommender API

This is the backend for the Career Recommendation System. It combines the Personality Model and the Reading Comprehension Model to provide a final career recommendation.

## Setup

1.  **Install Python 3.9+**
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the API

Run the following command in this directory:
```bash
python api.py
```
The server will start at `http://localhost:8000`.

## API Documentation

Once the server is running, go to `http://localhost:8000/docs` to see the interactive Swagger UI.

### Endpoint: `POST /predict`

**Request JSON Format:**
```json
{
  "personality_scores": {
    "R": 0.5, "I": 0.8, "A": 0.4, "S": 0.3, "E": 0.6, "C": 0.7,
    "O": 0.6, "C2": 0.7, "E2": 0.5, "A2": 0.6, "N": 0.2
  },
  "reading_responses": {
    "Qpattern1": 4,
    "Qpattern2": 5,
    "Qprobsolve1": 2,
    "Qprobsolve2": 3,
    "Qmgmt1": 1,
    "Qmgmt2": 2,
    "chosenActivity": "Discovering hidden trends or logical patterns",
    "freeText": "I enjoyed finding the patterns...",
    "chosenProject": "Analyzing data...",
    "team_choice": "Observation and analysis team"
  }
}
```

**Response JSON Format:**
```json
{
  "status": "success",
  "recommendation": "Data Science",
  "confidence_scores": {
    "Data Science": 0.65,
    "Software Development": 0.25,
    "Tech Project Management": 0.10
  },
  "details": { ... }
}
```
