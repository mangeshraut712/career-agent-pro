"""
Streaming Router for CareerAgentPro
=====================================
Server-Sent Events (SSE) endpoints for real-time AI responses.
Uses sse-starlette for proper SSE support.

Usage:
  GET /api/stream/analyze?job_description=...&resume_data=...
  GET /api/stream/cover-letter?job_description=...&resume_data=...
  GET /api/stream/chat?message=...&context=...

All endpoints return text/event-stream with:
  - data: partial response chunks
  - event: status updates (started, thinking, done)
"""

from __future__ import annotations

import json
import os
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Query
from sse_starlette.sse import EventSourceResponse
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/stream", tags=["streaming"])


def _get_client() -> AsyncOpenAI:
    """Get OpenRouter client."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="AI service not configured")
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=60.0,
    )


def _get_model() -> str:
    return os.getenv("STREAMING_MODEL", "google/gemini-2.0-flash-exp:free")


async def _stream_llm(
    system_prompt: str,
    user_prompt: str,
) -> AsyncGenerator[dict, None]:
    """Core streaming generator — yields SSE event dicts for EventSourceResponse."""
    client = _get_client()

    yield {"event": "started", "data": json.dumps({"status": "streaming"})}

    try:
        stream = await client.chat.completions.create(
            model=_get_model(),
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield {"data": json.dumps({"content": delta.content})}

        yield {"event": "done", "data": json.dumps({"status": "complete"})}

    except Exception as e:
        logger.error(f"Streaming failed: {e}", exc_info=True)
        yield {"event": "error", "data": json.dumps({"error": str(e)})}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/analyze")
async def stream_job_analysis(
    job_description: str = Query(..., min_length=10),
    resume_data: str = Query("{}", min_length=2),
):
    """
    Stream real-time job fit analysis.

    Frontend usage:
      const source = new EventSource('/api/stream/analyze?job_description=...&resume_data=...');
      source.onmessage = (e) => appendToUI(JSON.parse(e.data).content);
    """
    system_prompt = """You are CareerAgentPro, an expert career coach.
Analyze the job fit and stream your reasoning process in real-time.
Be specific, honest, and actionable. Structure your response with:
1. Fit Score (0-100) with justification
2. Top Strengths (3-5 matching points)
3. Gaps to Address (3-5 weaknesses)
4. Actionable Recommendations (what to do about each gap)
5. ATS Optimization Tips"""

    user_prompt = f"## Job Description\n{job_description}\n\n## Resume\n{resume_data}"

    return EventSourceResponse(
        _stream_llm(system_prompt, user_prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/cover-letter")
async def stream_cover_letter(
    job_description: str = Query(..., min_length=10),
    resume_data: str = Query("{}", min_length=2),
    template_type: str = Query("professional"),
):
    """
    Stream a cover letter being written in real-time.
    The user watches the letter being composed word by word.
    """
    system_prompt = f"""You are an expert cover letter writer.
Write a {template_type} cover letter for this application.
Requirements:
- Open with a specific hook about the company/role
- 2-3 body paragraphs connecting experience to requirements
- Strong closing with call-to-action
- Under 400 words
- No generic filler — every sentence must add value"""

    user_prompt = f"## Job\n{job_description}\n\n## Resume\n{resume_data}"

    return EventSourceResponse(
        _stream_llm(system_prompt, user_prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chat")
async def stream_career_chat(
    message: str = Query(..., min_length=1),
    context: str = Query(""),
):
    """
    Stream a career coaching conversation.
    Real-time chat interface for career advice.
    """
    system_prompt = """You are a senior career coach with 20+ years of experience.
Give direct, actionable advice. Don't hedge or qualify everything.
If something is a bad idea, say so. If something will work, explain why.
Use frameworks and specific examples."""

    user_prompt = f"{context}\n\nUser question: {message}" if context else message

    return EventSourceResponse(
        _stream_llm(system_prompt, user_prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/tailor-resume")
async def stream_resume_tailoring(
    job_description: str = Query(..., min_length=10),
    resume_data: str = Query("{}", min_length=2),
):
    """
    Stream resume tailoring suggestions in real-time.
    The coach talks through each section as they analyze.
    """
    system_prompt = """You are an ATS optimization expert and resume writer.
Walk through the resume section by section, providing tailoring suggestions.
Think out loud — show your reasoning process.
Format each suggestion as:
- Section: [section name]
- Current: [what's there]
- Suggestion: [what to change and why]
- Priority: [high/medium/low]"""

    user_prompt = f"## Target Job\n{job_description}\n\n## Current Resume\n{resume_data}"

    return EventSourceResponse(
        _stream_llm(system_prompt, user_prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
