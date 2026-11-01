// Centralized storage keys to ensure consistency across all pages
// NOTE: Do NOT add 'cap_' prefix here - secureGet/secureSet from secureStorage.ts adds it automatically!
export const STORAGE_KEYS = {
  // Profile data
  PROFILE: "profile",

  // Jobs data
  ANALYZED_JOBS: "analyzedJobs",
  CURRENT_JOB_FOR_RESUME: "currentJobForResume",

  // Resume data
  ENHANCEMENT_RESULT: "enhancementResult",

  // Interview data
  MOCK_INTERVIEW_DATA: "mockInterviewData",

  // Bullet library
  BULLET_LIBRARY: "bulletLibrary",
} as const;
