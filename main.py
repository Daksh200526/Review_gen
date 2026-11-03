from pathlib import Path
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()
API_KEY = os.getenv("API_KEY", "").strip()
AI_API_URL = os.getenv("AI_API_URL", "").strip()

app = FastAPI()

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
    return FileResponse(Path(__file__).with_name("review_gen.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate-review")
async def generate_review(data: ReviewData):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="API_KEY is not configured")
    if not AI_API_URL.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=503,
            detail="AI_API_URL must be configured with an http:// or https:// URL",
        )

    prompt = (
        f"Write a clinic review. Treatment: {data.treatment}. "
        f"Doctor: {data.doctor}. Experience: {', '.join(data.experience)}."
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                AI_API_URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"prompt": prompt},
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"AI service returned HTTP {exc.response.status_code}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="Unable to reach AI service") from exc

    try:
        result = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="AI service returned invalid JSON") from exc

    if isinstance(result, list) and result and isinstance(result[0], dict):
        result = result[0]
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="AI service returned an unsupported JSON response")

    review = result.get("generated_text") or result.get("text")
    if not review and isinstance(result.get("choices"), list) and result["choices"]:
        choice = result["choices"][0]
        if isinstance(choice, dict):
            message = choice.get("message", {})
        else:
            message = {}
        if isinstance(message, dict):
            review = message.get("content")
    if not isinstance(review, str) or not review.strip():
        raise HTTPException(
            status_code=502,
            detail=(
                "AI service response did not contain generated_text, text, "
                "or choices[0].message.content"
            ),
        )

    return {"review": review.strip()}
