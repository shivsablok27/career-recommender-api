from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from ensemble_predictor import EnsemblePredictor
import uvicorn

# Initialize App
app = FastAPI(
    title="Career Recommender API",
    description="API for Ensemble Career Recommendation System (Personality + Reading Comprehension)",
    version="1.0.0"
)

# Initialize Predictor (Loads models on startup)
predictor = EnsemblePredictor()

# --- Input Schemas ---
class PersonalityScores(BaseModel):
    R: float
    I: float
    A: float
    S: float
    E: float
    C: float
    O: float
    C2: float
    E2: float
    A2: float
    N: float

class ReadingResponses(BaseModel):
    Qpattern1: int
    Qpattern2: int
    Qprobsolve1: int
    Qprobsolve2: int
    Qmgmt1: int
    Qmgmt2: int
    chosenActivity: str
    freeText: str
    chosenProject: Optional[str] = ""
    team_choice: Optional[str] = ""

class PredictionRequest(BaseModel):
    personality_scores: PersonalityScores
    reading_responses: ReadingResponses

    class Config:
        json_schema_extra = {
            "example": {
                "personality_scores": {
                    "R": 0.60, "I": 0.95, "A": 0.40, "S": 0.30, "E": 0.40, "C": 0.80,
                    "O": 0.90, "C2": 0.85, "E2": 0.40, "A2": 0.50, "N": 0.30
                },
                "reading_responses": {
                    "Qpattern1": 5, "Qpattern2": 5,
                    "Qprobsolve1": 3, "Qprobsolve2": 3,
                    "Qmgmt1": 2, "Qmgmt2": 2,
                    "chosenActivity": "Discovering hidden trends or logical patterns",
                    "freeText": "I enjoy finding hidden patterns in data...",
                    "chosenProject": "Analyzing data to understand behaviour or predict outcomes",
                    "team_choice": "Observation and analysis team"
                }
            }
        }

# --- Endpoints ---

@app.get("/")
def home():
    return {"message": "Career Recommender API is running. Use /predict to get recommendations."}

@app.post("/predict")
def predict_career(request: PredictionRequest):
    try:
        # Convert Pydantic models to dicts
        p_data = request.personality_scores.dict()
        r_data = request.reading_responses.dict()
        
        # Run Prediction
        result = predictor.predict(p_data, r_data)
        
        return {
            "status": "success",
            "recommendation": result['final_recommendation'],
            "confidence_scores": result['final_scores'],
            "details": {
                "personality_contribution": result['personality_scores'],
                "reading_contribution": result['reading_scores']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run with: python api.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
