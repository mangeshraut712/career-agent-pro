import axios from "axios";

import type { ProfileData } from "@/types/shared";

interface ResumeExportPayload {
  name: string;
  email: string;
  phone?: string;
  linkedin?: string;
  summary: string;
  skills: string[];
  experience: Array<{
    role: string;
    company: string;
    description: string;
    duration: string;
  }>;
  education: Array<{
    institution: string;
    degree: string;
  }>;
}

const EXPORT_TIMEOUT_MS = 30_000;
const RESUME_EXPORT_LIMITS = {
  name: 200,
  email: 320,
  phone: 64,
  linkedin: 2_048,
  summary: 10_000,
  skills: 100,
  skill: 200,
  experience: 50,
  role: 300,
  company: 300,
  description: 10_000,
  duration: 100,
  education: 50,
  institution: 500,
  degree: 500,
} as const;
const EXPORTS = {
  pdf: {
    contentType: "application/pdf",
    extension: "pdf",
    label: "PDF",
  },
  docx: {
    contentType:
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    extension: "docx",
    label: "DOCX",
  },
} as const;

type ResumeExportFormat = keyof typeof EXPORTS;

export function toResumeExportPayload(
  profile: ProfileData,
): ResumeExportPayload {
  const name = profile.name.trim();
  const email = profile.email.trim();
  if (!name || !email) {
    throw new Error("Resume export requires a name and email");
  }

  const payload = {
    name,
    email,
    phone: profile.phone?.trim() || undefined,
    linkedin: profile.linkedin?.trim() || undefined,
    summary: profile.summary?.trim() || "",
    skills: (profile.skills || []).map((skill) => skill.trim()).filter(Boolean),
    experience: (profile.experience || []).map((experience) => ({
      role: experience.role.trim(),
      company: experience.company.trim(),
      description: experience.description.trim(),
      duration: [experience.startDate, experience.endDate]
        .filter(Boolean)
        .join(" - "),
    })),
    education: (profile.education || []).map((education) => ({
      institution: education.institution.trim(),
      degree: education.degree.trim(),
    })),
  };

  assertResumeExportBounds(payload);
  return payload;
}

function assertResumeExportBounds(payload: ResumeExportPayload): void {
  const tooLarge = () => {
    throw new Error("Resume export data is too large");
  };
  const hasExceededLimit = (value: string | undefined, limit: number) =>
    (value?.length || 0) > limit;

  if (
    hasExceededLimit(payload.name, RESUME_EXPORT_LIMITS.name) ||
    hasExceededLimit(payload.email, RESUME_EXPORT_LIMITS.email) ||
    hasExceededLimit(payload.phone, RESUME_EXPORT_LIMITS.phone) ||
    hasExceededLimit(payload.linkedin, RESUME_EXPORT_LIMITS.linkedin) ||
    hasExceededLimit(payload.summary, RESUME_EXPORT_LIMITS.summary) ||
    payload.skills.length > RESUME_EXPORT_LIMITS.skills ||
    payload.experience.length > RESUME_EXPORT_LIMITS.experience ||
    payload.education.length > RESUME_EXPORT_LIMITS.education
  ) {
    tooLarge();
  }

  if (
    payload.skills.some((skill) =>
      hasExceededLimit(skill, RESUME_EXPORT_LIMITS.skill),
    )
  ) {
    tooLarge();
  }

  if (
    payload.experience.some(
      (experience) =>
        hasExceededLimit(experience.role, RESUME_EXPORT_LIMITS.role) ||
        hasExceededLimit(experience.company, RESUME_EXPORT_LIMITS.company) ||
        hasExceededLimit(
          experience.description,
          RESUME_EXPORT_LIMITS.description,
        ) ||
        hasExceededLimit(experience.duration, RESUME_EXPORT_LIMITS.duration),
    )
  ) {
    tooLarge();
  }

  if (
    payload.education.some(
      (education) =>
        hasExceededLimit(
          education.institution,
          RESUME_EXPORT_LIMITS.institution,
        ) || hasExceededLimit(education.degree, RESUME_EXPORT_LIMITS.degree),
    )
  ) {
    tooLarge();
  }
}

export async function exportResumePdf(
  apiUrl: string,
  profile: ProfileData,
): Promise<void> {
  await exportResume(apiUrl, profile, "pdf");
}

export async function exportResumeDocx(
  apiUrl: string,
  profile: ProfileData,
): Promise<void> {
  await exportResume(apiUrl, profile, "docx");
}

async function exportResume(
  apiUrl: string,
  profile: ProfileData,
  format: ResumeExportFormat,
): Promise<void> {
  const exportConfig = EXPORTS[format];
  const response = await axios.post<Blob>(
    `${apiUrl}/export/${format}`,
    toResumeExportPayload(profile),
    {
      responseType: "blob",
      timeout: EXPORT_TIMEOUT_MS,
      headers: { Accept: exportConfig.contentType },
    },
  );
  const contentType = String(
    response.headers["content-type"] || "",
  ).toLowerCase();
  if (
    !(response.data instanceof Blob) ||
    response.data.size === 0 ||
    !contentType.startsWith(exportConfig.contentType)
  ) {
    throw new Error(
      `${exportConfig.label} export returned an unexpected response`,
    );
  }

  const objectUrl = URL.createObjectURL(response.data);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = exportFilename(
    response.headers["content-disposition"],
    exportConfig.extension,
  );
  link.rel = "noopener";
  try {
    document.body.appendChild(link);
    link.click();
  } finally {
    link.remove();
    URL.revokeObjectURL(objectUrl);
  }
}

function exportFilename(
  contentDisposition: unknown,
  extension: string,
): string {
  const header =
    typeof contentDisposition === "string" ? contentDisposition : "";
  const match = /\bfilename="?([^";\r\n]+)"?/i.exec(header);
  const candidate = match?.[1] || `resume.${extension}`;
  const safe = candidate.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^\.+/, "");
  if (!safe) {
    return `resume.${extension}`;
  }
  return safe.toLowerCase().endsWith(`.${extension}`)
    ? safe
    : `${safe}.${extension}`;
}
