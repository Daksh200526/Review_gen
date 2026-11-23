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
        "Write a short, first-person review that sounds like an ordinary person speaking naturally. "
        "Make this review feel fresh: vary its opening, sentence structure, and phrasing each time. Do not use a fixed template or stock review phrases. "
        "Keep the wording casual and straightforward, not overly polished, while still clear and readable. "
        f"The patient had {data.treatment} with dentist {data.doctor} at Beyond Dental in Vasanth Nagar. "
        f"The patient chose these experience points: {', '.join(data.experience)}. "
        "Use plain, everyday English and short, clear sentences. Keep the tone warm and honest, like a simple message to a friend. "
        "Avoid slang, fancy or formal words, idioms, marketing language, exaggerated praise, generic filler, and overly polished wording. "
        "Only mention details supported by the treatment, doctor, and selected experience points. Do not invent events, results, feelings, or claims. "
        "Include Vasanth Nagar in every review in a natural, unforced way. Mention the dentist by name naturally. "
        "Write 2 or 3 concise sentences. Do not use quotation marks, a heading, or a preamble; return only the review."
        "Instead of term Orthodontics include other words like teeth alignment, braces, teeth straightening, dental alignment, smile correction, dental braces, teeth adjustment, dental correction, smile enhancement, teeth positioning, dental realignment, smile improvement, teeth reshaping, bite correction, teeth alignment treatment."
        "Instead of term Dental Implants include other words like tooth replacement, artificial teeth, tooth restoration, dental fixtures, tooth implants, dental reconstruction, tooth support, dental restoration, oral prosthetics."
    )

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Help people express their own experience in plain, natural English. Sound conversational and sincere. Do not add facts or embellish their experience."},
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
