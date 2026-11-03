import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

app = FastAPI()

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

    prompt = (
        f"Write a genuine, positive, and natural first-person Google review for a clinic. "
        f"The patient underwent {data.treatment} with {data.doctor}. "
        f"Key positive aspects they experienced: {', '.join(data.experience)}. "
        f"Keep it under 3-4 sentences. Do not use quotation marks, placeholders, or preamble; provide only the review text."
    )

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You write short, natural Google reviews for clinic patients."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            res_data = response.json()
            review_text = res_data["choices"][0]["message"]["content"].strip()
            return {"review": review_text}
    except httpx.HTTPStatusError as e:
        error_details = e.response.text
        print(f"OpenAI API Error: {error_details}")
        raise HTTPException(status_code=e.response.status_code, detail=f"OpenAI API error: {error_details}")
    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))