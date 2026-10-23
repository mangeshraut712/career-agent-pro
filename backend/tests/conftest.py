"""Test bootstrap. Set before the application imports the rate limiter."""
import os

os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "10000")


def collect_route_paths(application) -> list[str]:
    """Include paths from routers that FastAPI keeps nested until request time."""
    paths: list[str] = []
    for route in application.routes:
        path = getattr(route, "path", None)
        if path:
            paths.append(path)
        original = getattr(route, "original_router", None)
        if original is None:
            continue
        for child in getattr(original, "routes", []):
            child_path = getattr(child, "path", None)
            if child_path:
                paths.append(child_path)
    return paths
