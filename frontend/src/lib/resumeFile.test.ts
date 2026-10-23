import { describe, expect, it } from "vitest";
import { MAX_RESUME_BYTES, validateResumeFile } from "@/lib/resumeFile";

describe("resume upload validation", () => {
    it("rejects legacy doc files, unknown types, and oversized uploads", () => {
        expect(validateResumeFile({ name: "resume.doc", size: 10 })).toMatch(/\.doc/);
        expect(validateResumeFile({ name: "payload.exe", size: 10 })).toMatch(/PDF, DOCX, or TXT/);
        expect(validateResumeFile({ name: "resume.pdf", size: MAX_RESUME_BYTES + 1 })).toMatch(/10MB/);
    });

    it("accepts a normal pdf", () => {
        expect(validateResumeFile({ name: "Resume.PDF", size: 1024 })).toBeNull();
    });
});
