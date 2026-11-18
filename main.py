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
        raise HTTPException(status_code=503, detail="OpenAI API key is not configured on the server.")

    prompt = (
        "Draft a short first-person review in a warm, conversational voice, based only on the patient's answers. "
        f"They had {data.treatment} with {data.doctor}. "
        f"They selected these parts of their experience: {', '.join(data.experience)}. "
        "Sound like a real person sharing a straightforward personal impression: use everyday wording, "
        "a natural rhythm, and vary the phrasing and sentence structure between reviews. "
        "Do not invent events, outcomes, feelings, or details that were not provided, and avoid generic promotional claims. "
        "Use correct, readable grammar, don't use high-end English, use simple relatable language; do not add deliberate mistakes. "
        "Write 2 to 4 concise sentences. Do not use quotation marks, placeholders, or a preamble; provide only the review text."
        "Mention the clinic location Vasanth Nagar to be placed in reviews and the way it is placed in the review should be very natural, realistic, highly trustworthy and should not feel forced. "
    )

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You help people express their own clinic experiences clearly and naturally. Keep every review grounded in the details they supplied."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            res_data = response.json()
    except httpx.HTTPStatusError as exc:
        try:
            error_message = exc.response.json().get("error", {}).get("message")
        except ValueError:
            error_message = None
        detail = f"OpenAI returned HTTP {exc.response.status_code}"
        if error_message:
            detail = f"{detail}: {error_message}"
        raise HTTPException(status_code=502, detail=detail) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="Could not connect to OpenAI.") from exc

    try:
        review_text = res_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="OpenAI returned an unexpected response.") from exc
    if not isinstance(review_text, str) or not review_text.strip():
        raise HTTPException(status_code=502, detail="OpenAI returned an empty review.")

    return {"review": review_text.strip()}
