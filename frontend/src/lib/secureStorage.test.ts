import { afterEach, describe, expect, it, vi } from "vitest";
import { sanitizeUrl, secureGet, secureSet } from "@/lib/secureStorage";

describe("secure storage expiry", () => {
    afterEach(() => {
        vi.useRealTimers();
        localStorage.clear();
    });

    it("keeps values that were saved without a TTL", () => {
        vi.useFakeTimers();
        vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));
        expect(secureSet("profile", { name: "Ada" })).toBe(true);

        vi.setSystemTime(new Date("2026-06-01T00:00:00Z"));
        expect(secureGet("profile")).toEqual({ name: "Ada" });
    });

    it("drops values after their TTL", () => {
        vi.useFakeTimers();
        vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));
        secureSet("draft", { ok: true }, 1_000);

        vi.advanceTimersByTime(1_001);
        expect(secureGet("draft")).toBeNull();
    });
});

describe("sanitizeUrl", () => {
    it("blocks scriptable and broken URLs", () => {
        expect(sanitizeUrl("javascript:alert(1)")).toBe("#");
        expect(sanitizeUrl("data:text/html,hi")).toBe("#");
        expect(sanitizeUrl('http://example.com" onclick="alert(1)')).toBe("#");
        expect(sanitizeUrl("http://example.com\njavascript:alert(1)")).toBe("#");
    });

    it("allows ordinary http(s) URLs", () => {
        expect(sanitizeUrl("https://www.linkedin.com/jobs/view/1")).toBe(
            "https://www.linkedin.com/jobs/view/1"
        );
    });
});
