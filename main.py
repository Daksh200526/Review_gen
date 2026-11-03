import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

app = FastAPI()

# Allow frontend requests from any origin (Render, Netlify, localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
        raise HTTPException(status_code=500, detail="API key is missing on the server.")

    # Tailored prompt instructing Gemini to write a natural, genuine 2-3 sentence patient review
    prompt = (
        f"Write a genuine, positive, and natural first-person Google review for a dental/medical clinic. "
        f"The patient underwent {data.treatment} with {data.doctor}. "
        f"Key positive aspects they experienced: {', '.join(data.experience)}. "
        f"Keep it under 3-4 sentences. Do not use quotation marks, placeholders, or preamble; provide only the review text."
    )

    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(gemini_url, json=payload)
            response.raise_for_status()
            res_data = response.json()

            # Extract generated review from Gemini's response schema
            generated_text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return {"review": generated_text}

    except httpx.HTTPStatusError as e:
        error_details = e.response.text
        print(f"Gemini API Error: {error_details}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Gemini API error: {error_details}")
    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))