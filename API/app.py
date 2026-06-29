import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

# Add AGENTS directory to system path to import decision_agent
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "AGENTS"))
from decision_agent import DecisionAgent

app = FastAPI(
    title="PII Detection and Governance API",
    description="Multi-agent service for cleaning, detecting, scoring risk, and governing Personally Identifiable Information.",
    version="1.0.0"
)

# Initialize DecisionAgent with paths
rules_path = os.path.join(os.path.dirname(__file__), "..", "RULES", "governance_rules.json")
decision_agent = DecisionAgent(rules_path=rules_path)

class EvaluateRequest(BaseModel):
    text: Optional[str] = None
    image_path: Optional[str] = None

class EvaluateResponse(BaseModel):
    decision: str
    risk_evaluation: dict
    governance_evaluation: dict
    detected_pii_count: int
    pii_items: List[dict]
    redacted_text: Optional[str] = None
    vision_metadata: dict

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "PII Detection and Governance API",
        "agents": ["preprocessing", "vision", "prompt", "risk", "governance", "decision"]
    }

@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest):
    if not request.text and not request.image_path:
        raise HTTPException(status_code=400, detail="Must provide either 'text' or 'image_path'.")
    
    payload = {}
    if request.text:
        payload["text"] = request.text
    if request.image_path:
        payload["image_path"] = request.image_path
        
    try:
        result = decision_agent.evaluate_payload(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
