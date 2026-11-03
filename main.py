from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import httpx

# Load the secret key from the .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

app = FastAPI()

# Allow your frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewData(BaseModel):
    treatment: str
    doctor: str
    experience: list[str]
@app.get("/")
async def root():
    return {"message": "Clinic Review API is running!"}
@app.post("/generate-review")
async def generate_review(data: ReviewData):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    
    # Construct your prompt
    prompt = f"Write a clinic review. Treatment: {data.treatment}. Doctor: {data.doctor}. Experience: {', '.join(data.experience)}."
    
    # Securely call your AI API from the backend
    # Example using a generic POST request structure
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "YOUR_AI_API_ENDPOINT_HERE",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"prompt": prompt} # Adjust payload to match your specific AI provider
            )
            response.raise_for_status()
            return {"review": response.json()["generated_text"]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))