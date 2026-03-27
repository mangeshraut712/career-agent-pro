"""
Tests for Main Application
============================
Verifies that all endpoints are registered and basic routing works.
"""

import pytest
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


class TestAppStartup:
    """Test that the FastAPI app starts correctly."""

    def test_app_exists(self):
        assert app is not None
        assert app.title == "CareerAgentPro API"

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "CareerAgentPro" in response.json()["message"]


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "CareerAgentPro"
        assert "version" in data
        assert "timestamp" in data


class TestCoreEndpointRegistration:
    """Verify all core endpoints are registered."""

    EXPECTED_ENDPOINTS = [
        "/parse-resume",
        "/extract-job",
        "/enhance-resume",
        "/generate-cover-letter",
        "/generate-communication",
        "/export/docx",
        "/export/pdf",
        "/export/latex",
        "/generate-autofill",
        "/assess-job-fit",
        "/analyze-bullets",
        "/detect-company-stage",
        "/analyze-complete-resume",
        "/bullet-library/select-for-job",
        "/verify-resume",
        "/verify-cover-letter",
        "/generate-outreach-strategy",
        "/verify-outreach",
        "/orchestrate-application",
        "/generate-elite-cover-letter",
        "/validate-bullet",
        "/assess-competencies",
        "/spin-text",
        "/verify-resume-quality",
        "/langchain/analyze-fit",
        "/langchain/tailor-resume",
        "/langchain/cover-letter",
        "/langchain/full-pipeline",
    ]

    def test_all_core_endpoints_registered(self):
        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        for endpoint in self.EXPECTED_ENDPOINTS:
            assert endpoint in routes, f"Endpoint {endpoint} not registered"

    def test_streaming_endpoints_registered(self):
        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        assert "/api/stream/analyze" in routes
        assert "/api/stream/cover-letter" in routes
        assert "/api/stream/chat" in routes
        assert "/api/stream/tailor-resume" in routes


class TestEndpointMethodValidation:
    """Verify endpoints reject wrong HTTP methods."""

    def test_post_endpoints_reject_get(self):
        post_only = [
            "/parse-resume",
            "/extract-job",
            "/enhance-resume",
            "/generate-cover-letter",
            "/assess-job-fit",
            "/langchain/analyze-fit",
        ]
        for endpoint in post_only:
            response = client.get(endpoint)
            assert response.status_code in (405, 422), \
                f"{endpoint} should reject GET, got {response.status_code}"

    def test_get_endpoints_reject_post(self):
        get_only = [
            "/health",
        ]
        for endpoint in get_only:
            response = client.post(endpoint, json={})
            assert response.status_code == 405, \
                f"{endpoint} should reject POST, got {response.status_code}"


class TestLangChainEndpointRegistration:
    """Verify LangChain-specific endpoints exist."""

    def test_langchain_analyze_fit(self):
        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        assert "/langchain/analyze-fit" in routes

    def test_langchain_tailor_resume(self):
        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        assert "/langchain/tailor-resume" in routes

    def test_langchain_cover_letter(self):
        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        assert "/langchain/cover-letter" in routes

    def test_langchain_full_pipeline(self):
        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        assert "/langchain/full-pipeline" in routes
