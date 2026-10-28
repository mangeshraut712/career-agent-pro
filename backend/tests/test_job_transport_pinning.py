"""Offline regressions for the untrusted job-URL transport boundary."""

import socket

import httpx
import pytest

from services.job_service import JobService, _MAX_REDIRECTS


class _Response:
    def __init__(self, status_code: int = 200, text: str = "<html>ok</html>", location: str | None = None):
        self.status_code = status_code
        self.text = text
        self.headers = {"location": location} if location else {}


def _recording_client(responses, seen, error: Exception | None = None):
    """Return an AsyncClient stand-in that cannot perform network I/O."""

    class FakeClient:
        def __init__(self, *args, **kwargs):
            seen["init"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers=None, extensions=None):
            seen.setdefault("requests", []).append({"url": url, "headers": headers or {}, "extensions": extensions})
            if error:
                raise error
            return responses.pop(0)

    return FakeClient


def _public_ipv4(host, port, *args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port))]


class TestPinnedJobUrlTransport:
    def setup_method(self):
        self.service = JobService(ai_service=None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "source_url",
        [
            "https://www.linkedin.com/jobs/view/9",
            "https://www.linkedin.com:443/jobs/view/9",
        ],
    )
    async def test_pins_ipv4_and_preserves_hostname_tls_context(self, monkeypatch, source_url):
        seen = {}
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
        monkeypatch.setattr("services.job_service.socket.getaddrinfo", _public_ipv4)
        monkeypatch.setattr(
            "services.job_service.httpx.AsyncClient",
            _recording_client([_Response()], seen),
        )

        text = await self.service._fetch_with_httpx(source_url)

        assert text == "<html>ok</html>"
        assert seen["init"] == {
            "follow_redirects": False,
            "timeout": 30.0,
            "trust_env": False,
            "verify": True,
        }
        assert seen["requests"] == [
            {
                "url": "https://8.8.8.8/jobs/view/9",
                "headers": {
                    "User-Agent": "Mozilla/5.0 (compatible; JobHelper/1.0)",
                    "Accept": "text/html",
                    "Host": "www.linkedin.com",
                },
                "extensions": {"sni_hostname": "www.linkedin.com"},
            }
        ]

    @pytest.mark.asyncio
    async def test_pins_ipv6_in_brackets_without_changing_host_or_sni(self, monkeypatch):
        seen = {}

        def public_ipv6(host, port, *args, **kwargs):
            return [
                (
                    socket.AF_INET6,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    ("2606:4700:4700::1111", port, 0, 0),
                )
            ]

        monkeypatch.setattr("services.job_service.socket.getaddrinfo", public_ipv6)
        monkeypatch.setattr(
            "services.job_service.httpx.AsyncClient",
            _recording_client([_Response()], seen),
        )

        text = await self.service._fetch_with_httpx("https://www.linkedin.com/jobs/view/9")

        assert text == "<html>ok</html>"
        request = seen["requests"][0]
        assert request["url"] == "https://[2606:4700:4700::1111]/jobs/view/9"
        assert request["headers"]["Host"] == "www.linkedin.com"
        assert request["extensions"] == {"sni_hostname": "www.linkedin.com"}

    @pytest.mark.asyncio
    async def test_dns_change_after_selection_cannot_reach_client_as_a_hostname(self, monkeypatch):
        seen = {"resolutions": []}

        def changing_dns(host, port, *args, **kwargs):
            seen["resolutions"].append((host, port))
            ip = "8.8.8.8" if len(seen["resolutions"]) == 1 else "127.0.0.1"
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

        monkeypatch.setattr("services.job_service.socket.getaddrinfo", changing_dns)
        monkeypatch.setattr(
            "services.job_service.httpx.AsyncClient",
            _recording_client([_Response()], seen),
        )

        text = await self.service._fetch_with_httpx("https://www.linkedin.com/jobs/view/9")

        assert text == "<html>ok</html>"
        assert seen["resolutions"] == [("www.linkedin.com", 443)]
        assert seen["requests"][0]["url"] == "https://8.8.8.8/jobs/view/9"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "answers",
        [
            [],
            [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
            ],
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("not-an-ip", 443))],
        ],
    )
    async def test_empty_mixed_or_invalid_answers_never_connect(self, monkeypatch, answers):
        def rejected_client(*args, **kwargs):
            raise AssertionError("rejected DNS results must not instantiate a client")

        monkeypatch.setattr("services.job_service.socket.getaddrinfo", lambda *args, **kwargs: answers)
        monkeypatch.setattr("services.job_service.httpx.AsyncClient", rejected_client)

        assert await self.service._fetch_with_httpx("https://www.linkedin.com/jobs/view/9") is None

    @pytest.mark.asyncio
    async def test_redirect_hops_are_resolved_and_pinned_independently(self, monkeypatch):
        seen = {"resolutions": []}

        def resolve(host, port, *args, **kwargs):
            seen["resolutions"].append((host, port))
            addresses = {
                "www.linkedin.com": "8.8.8.8",
                "jobs.lever.co": "1.1.1.1",
            }
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addresses[host], port))]

        monkeypatch.setattr("services.job_service.socket.getaddrinfo", resolve)
        monkeypatch.setattr(
            "services.job_service.httpx.AsyncClient",
            _recording_client(
                [
                    _Response(302, location="https://jobs.lever.co/acme/1"),
                    _Response(200, "<html>Engineer</html>"),
                ],
                seen,
            ),
        )

        text = await self.service._fetch_with_httpx("https://www.linkedin.com/jobs/view/9")

        assert text == "<html>Engineer</html>"
        assert seen["resolutions"] == [
            ("www.linkedin.com", 443),
            ("jobs.lever.co", 443),
        ]
        assert [request["url"] for request in seen["requests"]] == [
            "https://8.8.8.8/jobs/view/9",
            "https://1.1.1.1/acme/1",
        ]
        assert [request["headers"]["Host"] for request in seen["requests"]] == [
            "www.linkedin.com",
            "jobs.lever.co",
        ]
        assert [request["extensions"] for request in seen["requests"]] == [
            {"sni_hostname": "www.linkedin.com"},
            {"sni_hostname": "jobs.lever.co"},
        ]

    @pytest.mark.asyncio
    async def test_private_redirect_answer_is_rejected_before_second_request(self, monkeypatch):
        seen = {"resolutions": []}

        def resolve(host, port, *args, **kwargs):
            seen["resolutions"].append((host, port))
            ip = "8.8.8.8" if host == "www.linkedin.com" else "127.0.0.1"
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

        monkeypatch.setattr("services.job_service.socket.getaddrinfo", resolve)
        monkeypatch.setattr(
            "services.job_service.httpx.AsyncClient",
            _recording_client([_Response(302, location="https://jobs.lever.co/acme/1")], seen),
        )

        text = await self.service._fetch_with_httpx("https://www.linkedin.com/jobs/view/9")

        assert text is None
        assert seen["resolutions"] == [
            ("www.linkedin.com", 443),
            ("jobs.lever.co", 443),
        ]
        assert [request["url"] for request in seen["requests"]] == ["https://8.8.8.8/jobs/view/9"]

    @pytest.mark.asyncio
    async def test_redirect_budget_stays_bounded_with_pinned_requests(self, monkeypatch):
        seen = {}
        monkeypatch.setattr("services.job_service.socket.getaddrinfo", _public_ipv4)
        monkeypatch.setattr(
            "services.job_service.httpx.AsyncClient",
            _recording_client(
                [_Response(302, location="/jobs/view/next")] * (_MAX_REDIRECTS + 1),
                seen,
            ),
        )

        text = await self.service._fetch_with_httpx("https://www.linkedin.com/jobs/view/9")

        assert text is None
        assert len(seen["requests"]) == _MAX_REDIRECTS + 1
        assert {request["url"] for request in seen["requests"]} == {
            "https://8.8.8.8/jobs/view/9",
            "https://8.8.8.8/jobs/view/next",
        }

    @pytest.mark.asyncio
    async def test_tls_failure_returns_none_without_falling_back_to_hostname(self, monkeypatch):
        seen = {}
        monkeypatch.setattr("services.job_service.socket.getaddrinfo", _public_ipv4)
        monkeypatch.setattr(
            "services.job_service.httpx.AsyncClient",
            _recording_client([], seen, httpx.ConnectError("TLS verification failed")),
        )

        text = await self.service._fetch_with_httpx("https://www.linkedin.com/jobs/view/9")

        assert text is None
        assert seen["requests"][0]["url"] == "https://8.8.8.8/jobs/view/9"
        assert seen["requests"][0]["extensions"] == {"sni_hostname": "www.linkedin.com"}

    @pytest.mark.asyncio
    async def test_rejected_input_never_resolves_or_connects(self, monkeypatch):
        def unexpected(*args, **kwargs):
            raise AssertionError("rejected input must not resolve or connect")

        monkeypatch.setattr("services.job_service.socket.getaddrinfo", unexpected)
        monkeypatch.setattr("services.job_service.httpx.AsyncClient", unexpected)

        assert await self.service._fetch_with_httpx("https://evil.example/job") is None
