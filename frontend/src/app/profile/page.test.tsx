import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { secureGet, secureSet } from "@/lib/secureStorage";
import { sanitizeProfile } from "./profileData";
import type { ProfileData } from "./profileTypes";
import ProfilePage from "./page";

const toast = vi.fn();
vi.mock("@/components/ui/Toast", () => ({
  useToast: () => ({ toast }),
}));

const uploadedLabel = "Upload resume as PDF, DOCX, or TXT";

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

function stubParse(data: Record<string, unknown>) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(completedParseStream(data)),
  );
}

async function uploadResume() {
  render(<ProfilePage />);
  const input = await screen.findByLabelText(uploadedLabel);
  fireEvent.change(input, {
    target: {
      files: [new File(["resume"], "resume.txt", { type: "text/plain" })],
    },
  });
}

function savedProfile(): ProfileData | null {
  return secureGet<ProfileData>("profile");
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

beforeEach(() => {
  toast.mockReset();
});

describe("sanitizeProfile", () => {
  it("drops scalar fields of the wrong type instead of passing them through", () => {
    const result = sanitizeProfile({ name: ["Ada", "Lovelace"], email: 42 });
    expect(result.name).toBe("");
    expect(result.email).toBe("");
  });

  it("keeps well-formed fields around malformed ones", () => {
    const result = sanitizeProfile({
      name: "Ada Lovelace",
      skills: ["Python", 7, null, "SQL"],
      certifications: ["AWS", { name: "not a string" }],
    });
    expect(result.name).toBe("Ada Lovelace");
    expect(result.skills).toEqual(["Python", "SQL"]);
    expect(result.certifications).toEqual(["AWS"]);
  });

  it("returns an empty profile for a non-object value", () => {
    expect(sanitizeProfile(null)).toMatchObject({ name: "", skills: [] });
    expect(sanitizeProfile("garbage")).toMatchObject({ name: "", skills: [] });
  });

  it("reads the title from either key models use", () => {
    expect(sanitizeProfile({ jobTitle: "Engineer" }).title).toBe("Engineer");
    expect(sanitizeProfile({ title: "Analyst" }).title).toBe("Analyst");
  });
});

describe("ProfilePage resume parse", () => {
  it("stores the parsed profile so a reload can read it", async () => {
    stubParse({
      name: "Ada Lovelace",
      email: "ada@example.com",
      skills: ["Python"],
      summary: "Analytical engines.",
    });

    await uploadResume();

    await waitFor(() => {
      const saved = savedProfile();
      expect(saved?.name).toBe("Ada Lovelace");
      expect(saved?.email).toBe("ada@example.com");
      expect(saved?.skills).toEqual(["Python"]);
    });
  });

  it("keeps saved projects and certifications when the resume lists none", async () => {
    secureSet("profile", {
      name: "",
      title: "",
      email: "",
      phone: "",
      location: "",
      linkedin: "",
      github: "",
      portfolio: "",
      jobSearchStatus: "actively_looking",
      experience: [],
      education: [],
      projects: [{ id: "p1", name: "Portfolio site", description: "" }],
      certifications: ["AWS Solutions Architect"],
      skills: [],
      preferredSkills: [],
      summary: "",
    });
    stubParse({ name: "Ada Lovelace", email: "ada@example.com" });

    await uploadResume();

    await waitFor(() => {
      const saved = savedProfile();
      expect(saved?.name).toBe("Ada Lovelace");
      // The hand-added entry survives the import.
      expect(saved?.projects).toEqual([
        expect.objectContaining({ id: "p1", name: "Portfolio site" }),
      ]);
      expect(saved?.certifications).toEqual(["AWS Solutions Architect"]);
    });
  });

  it("does not save a name of the wrong type, and the page still renders after reload", async () => {
    secureSet("profile", { ...sanitizeProfile({}), name: "Ada Lovelace" });
    stubParse({ name: ["Ada", "Lovelace"], email: "ada@example.com" });

    await uploadResume();

    await waitFor(() => {
      const saved = savedProfile();
      expect(saved?.name).toBe("Ada Lovelace");
      expect(saved?.email).toBe("ada@example.com");
    });
  });

  it("still opens when storage already holds a malformed record", async () => {
    secureSet("profile", { name: ["Ada", "Lovelace"], skills: "Python" });

    render(<ProfilePage />);

    expect(await screen.findByLabelText(uploadedLabel)).toBeInTheDocument();
  });

  it("tells the user when the browser refuses to save", async () => {
    // secureSet leaves an existing key untouched when it fails, so this test
    // only means something if storage is empty to begin with. setup.ts clears
    // it globally; assert that here so losing that clear fails loudly.
    expect(savedProfile()).toBeNull();

    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("QuotaExceededError");
    });
    stubParse({ name: "Ada Lovelace" });

    await uploadResume();

    await waitFor(() => {
      expect(toast).toHaveBeenCalledWith(
        expect.stringContaining("will not survive a reload"),
        "error",
      );
    });
    expect(toast).not.toHaveBeenCalledWith(
      "Profile Synchronization Complete!",
      "success",
    );
    expect(savedProfile()).toBeNull();
  });

  it("reports success only when the profile was actually saved", async () => {
    stubParse({ name: "Ada Lovelace" });

    await uploadResume();

    await waitFor(() => {
      expect(toast).toHaveBeenCalledWith(
        "Profile Synchronization Complete!",
        "success",
      );
    });
    expect(savedProfile()?.name).toBe("Ada Lovelace");
  });
});
