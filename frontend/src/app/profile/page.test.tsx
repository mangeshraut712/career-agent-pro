import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { secureGet } from "@/lib/secureStorage";

import ProfilePage from "./page";

vi.mock("@/components/ui/Toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

function completedParseStream(data: Record<string, unknown>) {
  const payload = `data: ${JSON.stringify({ status: "completed", data })}\n\n`;
  const encoded = new TextEncoder().encode(payload);
  let sent = false;
  return {
    ok: true,
    body: {
      getReader: () => ({
        read: async () => {
          if (sent) {
            return { done: true, value: undefined };
          }
          sent = true;
          return { done: false, value: encoded };
        },
      }),
    },
  };
}

describe("ProfilePage resume parse", () => {
  it("stores the parsed profile so a reload can read it", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        completedParseStream({
          name: "Ada Lovelace",
          email: "ada@example.com",
          skills: ["Python"],
          summary: "Analytical engines.",
        }),
      ),
    );

    render(<ProfilePage />);
    const input = await screen.findByLabelText(
      "Upload resume as PDF, DOCX, or TXT",
    );
    fireEvent.change(input, {
      target: {
        files: [new File(["Ada Lovelace"], "resume.txt", { type: "text/plain" })],
      },
    });

    await waitFor(() => {
      const saved = secureGet<{ name: string; email: string; skills: string[] }>(
        "profile",
      );
      expect(saved?.name).toBe("Ada Lovelace");
      expect(saved?.email).toBe("ada@example.com");
      expect(saved?.skills).toEqual(["Python"]);
    });
  });
});
