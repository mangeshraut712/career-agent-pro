export const JOB_STATUSES = [
  "analyzing",
  "resume_updated",
  "applied",
  "interviewing",
  "offer",
  "rejected",
] as const;

export type JobStatus = (typeof JOB_STATUSES)[number];

export interface StoredAnalyzedJob {
  id?: string;
  title?: string;
  company?: string;
  url?: string;
  status?: string;
  analyzedAt?: string;
  createdAt?: string;
  description?: string;
  skills?: string[];
  [key: string]: unknown;
}

export interface TrackedJobView {
  id: string;
  title: string;
  company: string;
  url: string;
  status: JobStatus;
  createdAt: string;
}

export function withStableIds(jobs: StoredAnalyzedJob[]): StoredAnalyzedJob[] {
  return jobs.map((job, index) =>
    job.id ? job : { ...job, id: `job-${index}` },
  );
}

export function isJobStatus(value: unknown): value is JobStatus {
  return (
    typeof value === "string" &&
    (JOB_STATUSES as readonly string[]).includes(value)
  );
}

/**
 * Display shape for the dashboard. Unknown statuses fall back to analyzing.
 * Ids stay stable so a later status update can find the same record.
 */
export function toTrackedJob(
  job: StoredAnalyzedJob,
  index: number,
): TrackedJobView {
  return {
    id: job.id || `job-${index}`,
    title: job.title || "Unknown Job",
    company: job.company || "Unknown Company",
    url: job.url || "",
    status: isJobStatus(job.status) ? job.status : "analyzing",
    createdAt: job.createdAt || job.analyzedAt || new Date(0).toISOString(),
  };
}

export function updateStoredJobStatus<T extends StoredAnalyzedJob>(
  jobs: T[],
  jobId: string,
  status: JobStatus,
): T[] {
  return jobs.map((job) => (job.id === jobId ? { ...job, status } : job));
}

/**
 * Put a freshly analyzed job at the top of the saved list, replacing any
 * earlier record for the same URL. The tracker status (applied,
 * interviewing, ...) belongs to the user, so a re-analysis keeps it.
 */
export function upsertAnalyzedJob<T extends { url?: string }>(
  jobs: T[],
  job: T,
): T[] {
  const previous = jobs.find((saved) => saved.url === job.url) as
    (T & { status?: unknown }) | undefined;
  const merged =
    previous && isJobStatus(previous.status)
      ? { ...job, status: previous.status }
      : job;
  return [merged, ...jobs.filter((saved) => saved.url !== job.url)];
}

export function removeStoredJob<T extends { id?: string }>(
  jobs: T[],
  jobId: string,
): T[] {
  return jobs.filter((job) => job.id !== jobId);
}
