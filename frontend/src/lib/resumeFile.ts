export const MAX_RESUME_BYTES = 10 * 1024 * 1024;
export const RESUME_EXTENSIONS = [".pdf", ".docx", ".txt"];

export function resumeExtension(filename: string): string {
    const dot = filename.lastIndexOf(".");
    return dot >= 0 ? filename.slice(dot).toLowerCase() : "";
}

export function validateResumeFile(file: { name: string; size: number }): string | null {
    const extension = resumeExtension(file.name);
    if (extension === ".doc") {
        return "Legacy .doc files are not supported. Save the resume as PDF, DOCX, or TXT.";
    }
    if (!RESUME_EXTENSIONS.includes(extension)) {
        return "Upload a PDF, DOCX, or TXT resume.";
    }
    if (file.size > MAX_RESUME_BYTES) {
        return "Resume must be 10MB or smaller.";
    }
    return null;
}
