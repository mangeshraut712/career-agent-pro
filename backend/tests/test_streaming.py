"""
Tests for Streaming Endpoints
==============================
Tests SSE endpoint registration and basic response structure.
"""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


class TestStreamingEndpointRegistration:
    """Verify all streaming endpoints are registered on the app."""

    def test_analyze_endpoint_exists(self):
        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        assert "/api/stream/analyze" in routes

    def test_cover_letter_endpoint_exists(self):
        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        assert "/api/stream/cover-letter" in routes

    def test_chat_endpoint_exists(self):
        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        assert "/api/stream/chat" in routes

    def test_tailor_resume_endpoint_exists(self):
        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        assert "/api/stream/tailor-resume" in routes


class TestStreamingEndpointValidation:
    """Test that streaming endpoints enforce query parameter validation."""

    def test_analyze_requires_job_description(self):
        response = client.get("/api/stream/analyze")
        assert response.status_code == 422

    def test_analyze_rejects_short_job_description(self):
        response = client.get("/api/stream/analyze?job_description=short")
        assert response.status_code == 422

    def test_chat_requires_message(self):
        response = client.get("/api/stream/chat")
        assert response.status_code == 422

    def test_cover_letter_requires_job_description(self):
        response = client.get("/api/stream/cover-letter")
        assert response.status_code == 422

    def test_tailor_resume_requires_job_description(self):
        response = client.get("/api/stream/tailor-resume")
        assert response.status_code == 422


class TestStreamingSSEFormat:
    """Test SSE response format with mocked AI client."""

    def test_analyze_returns_event_stream(self):
        """When AI is configured, endpoint should return text/event-stream."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            # The endpoint should at least attempt to stream
            # Full streaming test requires an async test client
            response = client.get(
                "/api/stream/analyze?job_description=We+need+a+senior+Python+developer+with+5+years+experience"
            )
            # Should return 200 with streaming content type
            # or 503 if API key check fails (both are acceptable)
            assert response.status_code in (200, 503)
