from pathlib import Path

import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY", "").strip()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewData(BaseModel):
    treatment: str
    doctor: str
    experience: list[str]

@app.get("/")
async def root():
    return FileResponse(Path(__file__).with_name("review_gen.html"))


@app.post("/generate-review")
async def generate_review(data: ReviewData):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="Gemini API key is not configured on the server.")

    prompt = (
        f"Write a genuine, positive, and natural first-person Google review for a clinic. "
        f"The patient underwent {data.treatment} with {data.doctor}. "
        f"Key positive aspects they experienced: {', '.join(data.experience)}. "
        f"Keep it under 3-4 sentences. Do not use quotation marks, placeholders, or preamble; provide only the review text."
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={API_KEY}"
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            res_data = response.json()
            review_text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return {"review": review_text}
    except httpx.HTTPStatusError as e:
        error_details = e.response.text
        print(f"Gemini API Error: {error_details}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Gemini API error: {error_details}")
    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))