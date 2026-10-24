"""Custom middleware for the FastAPI application."""
import os
import time
import logging
from typing import Callable
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers, MutableHeaders
from collections import defaultdict
from datetime import datetime, timedelta

from security_utils import body_size_status

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[datetime]] = defaultdict(list)
        self.cleanup_interval = timedelta(minutes=5)
        self.last_cleanup = datetime.now()
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed based on rate limit."""
        now = datetime.now()
        
        # Cleanup old entries periodically
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup(now)
            self.last_cleanup = now
        
        # Remove requests older than 1 minute
        cutoff = now - timedelta(minutes=1)
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > cutoff
        ]
        
        # Check if limit exceeded
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        
        # Record this request
        self.requests[client_id].append(now)
        return True
    
    def _cleanup(self, now: datetime):
        """Remove old entries to prevent memory leaks."""
        cutoff = now - timedelta(minutes=2)
        keys_to_remove = [
            key for key, times in self.requests.items()
            if not any(t > cutoff for t in times)
        ]
        for key in keys_to_remove:
            del self.requests[key]


def _configured_rate_limit() -> int:
    raw = os.getenv("RATE_LIMIT_PER_MINUTE", "60")
    try:
        return max(1, int(raw))
    except ValueError:
        return 60


# Global rate limiter instance
rate_limiter = RateLimiter(_configured_rate_limit())


class RateLimitMiddleware:
    """ASGI rate limit that does not buffer streaming response bodies."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")
        headers = Headers(scope=scope)
        size_status = body_size_status(headers.get("content-length"))
        if size_status is not None:
            message = (
                "Request body is too large."
                if size_status == 413
                else "Content-Length is invalid."
            )
            response = JSONResponse(
                status_code=size_status,
                content={"error": message},
            )
            await response(scope, receive, send)
            return

        if method == "OPTIONS" or path in {"/", "/health"}:
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        client_id = client[0] if client else "unknown"
        if not rate_limiter.is_allowed(client_id):
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                },
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


class SecurityHeadersMiddleware:
    """Add response security headers without buffering the body."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                headers["X-Frame-Options"] = "DENY"
                headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
                headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
            await send(message)

        await self.app(scope, receive, send_with_headers)


async def rate_limit_middleware(request: Request, call_next: Callable) -> Response:
    """Rate limiting middleware."""
    # Skip rate limiting for health checks
    if request.url.path == "/health" or request.url.path == "/":
        return await call_next(request)
    
    # Get client identifier
    client_id = request.client.host if request.client else "unknown"
    
    # Check rate limit
    if not rate_limiter.is_allowed(client_id):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
            },
        )
    
    return await call_next(request)


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """Logging middleware for request/response."""
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown",
        },
    )
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time": round(process_time, 3),
            },
        )
        
        # Add process time header
        response.headers["X-Process-Time"] = str(round(process_time, 3))
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            "Request failed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "error": str(e),
                "process_time": round(process_time, 3),
            },
            exc_info=True,
        )
        raise


async def security_headers_middleware(request: Request, call_next: Callable) -> Response:
    """Add security headers to responses."""
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response

