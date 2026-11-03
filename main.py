from pathlib import Path

import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
    return FileResponse(Path(__file__).with_name("review_gen.html"))

@app.post("/generate-review")
async def generate_review(data: ReviewData):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key is missing on the server.")

    prompt = (
        f"Write a genuine, positive, and natural first-person Google review for a dental/medical clinic. "
        f"The patient underwent {data.treatment} with {data.doctor}. "
        f"Key positive aspects they experienced: {', '.join(data.experience)}. "
        f"Keep it under 3-4 sentences. Do not use quotation marks, placeholders, or preamble; provide only the review text."
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    # Sequence of models to try in order
    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]

    async with httpx.AsyncClient(timeout=30.0) as client:
        last_error = ""
        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
            try:
                response = await client.post(url, json=payload)
                if response.status_code == 200:
                    res_data = response.json()
                    review_text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return {"review": review_text}
                else:
                    last_error = f"{model} returned {response.status_code}: {response.text}"
                    print(f"Model failed: {last_error}")
            except Exception as e:
                last_error = f"{model} error: {str(e)}"
                print(last_error)

        # If all fail, throw the exact detail
        raise HTTPException(status_code=500, detail=f"All models failed. Last detail: {last_error}")