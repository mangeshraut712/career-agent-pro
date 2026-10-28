import axios from "axios";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { exportResumeDocx, exportResumePdf } from "./resumeExport";

vi.mock("axios", () => ({
  default: { post: vi.fn() },
}));

const post = vi.mocked(axios.post);
const profile = {
  name: "Ada Lovelace",
  email: "ada@example.com",
  phone: "555-0100",
  linkedin: " https://www.linkedin.com/in/ada-lovelace ",
  summary: "Built analytical engines.",
  skills: ["Python", "Mathematics"],
  experience: [
    {
      role: "Engineer",
      company: "Analytical Engines",
      description: "Designed computation workflows.",
      startDate: "1842",
      endDate: "1843",
    },
  ],
  education: [{ institution: "Drexel", degree: "MS" }],
};

describe("exportResumePdf", () => {
  const createObjectURL = vi.fn(() => "blob:resume");
  const revokeObjectURL = vi.fn();

  beforeEach(() => {
    post.mockReset();
    createObjectURL.mockClear();
    revokeObjectURL.mockClear();
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: createObjectURL,
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: revokeObjectURL,
    });
  });

  it("posts a normalized profile and downloads the verified PDF response", async () => {
    post.mockResolvedValue({
      data: new Blob(["%PDF-1.7"], { type: "application/pdf" }),
      headers: {
        "content-type": "application/pdf",
        "content-disposition": 'attachment; filename="Ada_resume.pdf"',
      },
    });
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);

    await exportResumePdf("https://api.example", profile);

    expect(post).toHaveBeenCalledWith(
      "https://api.example/export/pdf",
      {
        name: "Ada Lovelace",
        email: "ada@example.com",
        phone: "555-0100",
        linkedin: "https://www.linkedin.com/in/ada-lovelace",
        summary: "Built analytical engines.",
        skills: ["Python", "Mathematics"],
        experience: [
          {
            role: "Engineer",
            company: "Analytical Engines",
            description: "Designed computation workflows.",
            duration: "1842 - 1843",
          },
        ],
        education: [{ institution: "Drexel", degree: "MS" }],
      },
      {
        responseType: "blob",
        timeout: 30_000,
        headers: { Accept: "application/pdf" },
      },
    );
    expect(click).toHaveBeenCalledOnce();
    expect(createObjectURL).toHaveBeenCalledOnce();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:resume");
    click.mockRestore();
  });

  it("does not download a non-PDF error response", async () => {
    post.mockResolvedValue({
      data: new Blob(['{"detail":"failed"}'], { type: "application/json" }),
      headers: { "content-type": "application/json" },
    });

    await expect(
      exportResumePdf("https://api.example", profile),
    ).rejects.toThrow("PDF export returned an unexpected response");
    expect(createObjectURL).not.toHaveBeenCalled();
  });

  it("uses the DOCX endpoint and rejects a response with the wrong media type", async () => {
    post.mockResolvedValue({
      data: new Blob(["not a document"], { type: "text/plain" }),
      headers: { "content-type": "text/plain" },
    });

    await expect(
      exportResumeDocx("https://api.example", profile),
    ).rejects.toThrow("DOCX export returned an unexpected response");
    expect(post).toHaveBeenCalledWith(
      "https://api.example/export/docx",
      expect.any(Object),
      expect.objectContaining({
        responseType: "blob",
        headers: {
          Accept:
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        },
      }),
    );
    expect(createObjectURL).not.toHaveBeenCalled();
  });

  it.each([
    ["PDF", exportResumePdf, "application/pdf-malformed"],
    [
      "DOCX",
      exportResumeDocx,
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document-malformed",
    ],
  ])(
    "rejects a spoofed %s content type",
    async (_format, exportResume, contentType) => {
      post.mockResolvedValue({
        data: new Blob(["not the requested format"]),
        headers: { "content-type": contentType },
      });
      const click = vi
        .spyOn(HTMLAnchorElement.prototype, "click")
        .mockImplementation(() => undefined);

      try {
        await expect(
          exportResume("https://api.example", profile),
        ).rejects.toThrow("export returned an unexpected response");
        expect(createObjectURL).not.toHaveBeenCalled();
      } finally {
        click.mockRestore();
      }
    },
  );

  it.each([
    ["PDF", exportResumePdf, "application/pdf; charset=binary"],
    [
      "DOCX",
      exportResumeDocx,
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document; charset=binary",
    ],
  ])(
    "accepts a parameterized %s content type",
    async (_format, exportResume, contentType) => {
      post.mockResolvedValue({
        data: new Blob(["valid export"]),
        headers: { "content-type": contentType },
      });
      const click = vi
        .spyOn(HTMLAnchorElement.prototype, "click")
        .mockImplementation(() => undefined);

      try {
        await expect(
          exportResume("https://api.example", profile),
        ).resolves.toBeUndefined();
        expect(createObjectURL).toHaveBeenCalledOnce();
      } finally {
        click.mockRestore();
      }
    },
  );

  it("rejects an incomplete profile before making a request", async () => {
    await expect(
      exportResumePdf("https://api.example", { ...profile, name: " " }),
    ).rejects.toThrow("name and email");
    expect(post).not.toHaveBeenCalled();
  });

  it("rejects oversized export data before making a request", async () => {
    await expect(
      exportResumePdf("https://api.example", {
        ...profile,
        skills: Array.from({ length: 101 }, () => "Python"),
      }),
    ).rejects.toThrow("Resume export data is too large");

    expect(post).not.toHaveBeenCalled();
  });

  it("rejects an oversized summary before making a request", async () => {
    await expect(
      exportResumePdf("https://api.example", {
        ...profile,
        summary: "x".repeat(10_001),
      }),
    ).rejects.toThrow("Resume export data is too large");

    expect(post).not.toHaveBeenCalled();
  });
});
