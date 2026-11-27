import type { Education, Experience, ProfileData, Project } from "./profileTypes";

/**
 * The parse endpoint returns model-generated JSON whose shape is not
 * guaranteed, and localStorage can hold a record written by an older build.
 * Both are treated as untrusted and checked field by field before they reach
 * state: a truthy non-string name (["Ada", "Lovelace"]) used to be saved and
 * then crash the header's .split(" ") on every reload.
 */

export const EMPTY_PROFILE: ProfileData = {
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
  projects: [],
  certifications: [],
  skills: [],
  preferredSkills: [],
  summary: "",
};

const newId = () => Math.random().toString(36).substr(2, 9);

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function asStringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

/** Records that are not plain objects are skipped, not coerced. */
function asRecords(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) return [];
  return value.filter(
    (item): item is Record<string, unknown> =>
      item !== null && typeof item === "object" && !Array.isArray(item),
  );
}

function asExperienceList(value: unknown): Experience[] {
  return asRecords(value).map((exp) => ({
    id: asString(exp.id) || newId(),
    company: asString(exp.company),
    role: asString(exp.role) || asString(exp.title),
    location: asString(exp.location),
    startDate: asString(exp.startDate),
    endDate: asString(exp.endDate),
    type: asString(exp.type) || "Full-time",
    description: asString(exp.description),
  }));
}

function asEducationList(value: unknown): Education[] {
  return asRecords(value).map((edu) => ({
    id: asString(edu.id) || newId(),
    institution: asString(edu.institution) || asString(edu.school),
    degree: asString(edu.degree),
    field: asString(edu.field) || asString(edu.major),
    startDate: asString(edu.startDate),
    endDate: asString(edu.endDate) || asString(edu.graduation_year) || asString(edu.year),
  }));
}

function asProjectList(value: unknown): Project[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item): Project[] => {
    // A bare string is a project with only a name.
    if (typeof item === "string") {
      return [{ id: newId(), name: item, description: "" }];
    }
    if (item === null || typeof item !== "object" || Array.isArray(item)) {
      return [];
    }
    const project = item as Record<string, unknown>;
    return [
      {
        id: asString(project.id) || newId(),
        name: asString(project.name),
        description: asString(project.description),
        technologies: asStringList(project.technologies),
      },
    ];
  });
}

/**
 * Coerce an untrusted value into a well-formed ProfileData. Fields of the
 * wrong type are dropped (treated as absent) rather than passed through, so
 * the result is always safe to render and to persist.
 */
export function sanitizeProfile(value: unknown): ProfileData {
  const raw: Record<string, unknown> =
    value !== null && typeof value === "object" && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : {};

  return {
    ...EMPTY_PROFILE,
    name: asString(raw.name),
    // Models return the headline under either key.
    title: asString(raw.title) || asString(raw.jobTitle),
    email: asString(raw.email),
    phone: asString(raw.phone),
    location: asString(raw.location),
    linkedin: asString(raw.linkedin),
    github: asString(raw.github),
    portfolio: asString(raw.portfolio) || asString(raw.website),
    jobSearchStatus: asString(raw.jobSearchStatus) || EMPTY_PROFILE.jobSearchStatus,
    experience: asExperienceList(raw.experience),
    education: asEducationList(raw.education),
    projects: asProjectList(raw.projects),
    certifications: asStringList(raw.certifications),
    skills: asStringList(raw.skills),
    preferredSkills: asStringList(raw.preferredSkills),
    summary: asString(raw.summary),
  };
}
