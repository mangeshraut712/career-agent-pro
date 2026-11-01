/**
 * Application-wide constants
 */

// App metadata
export const APP_NAME = "CareerAgentPro";
export const APP_VERSION = "1.1.0";
export const APP_DESCRIPTION = "AI-Powered Career Platform";

// API configuration
export const API_TIMEOUT = 60000; // 60 seconds
export const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
export const ALLOWED_FILE_TYPES = [".pdf", ".doc", ".docx", ".txt"];

// UI constants
export const TOAST_DURATION = 3000; // 3 seconds
export const DEBOUNCE_DELAY = 300; // 300ms
export const ANIMATION_DURATION = 600; // 600ms

// Pagination
export const ITEMS_PER_PAGE = 10;
export const MAX_ITEMS_TO_SHOW = 100;

// Validation
export const MIN_PASSWORD_LENGTH = 8;
export const MAX_NAME_LENGTH = 100;
export const MAX_DESCRIPTION_LENGTH = 1000;
export const MAX_BULLET_POINTS = 10;

// Job fit scoring
export const FIT_SCORE_EXCELLENT = 80;
export const FIT_SCORE_GOOD = 60;
export const FIT_SCORE_FAIR = 40;

// Resume sections
export const RESUME_SECTIONS = [
  "summary",
  "experience",
  "education",
  "skills",
  "certifications",
  "projects",
] as const;

// Company stages
export const COMPANY_STAGES = [
  "Early Stage",
  "Growth Stage",
  "Enterprise",
] as const;

// Communication types
export const COMMUNICATION_TYPES = [
  "cover_letter",
  "email",
  "linkedin",
  "follow_up",
] as const;

// Export formats
export const EXPORT_FORMATS = ["pdf", "docx", "latex"] as const;

// Routes
export const ROUTES = {
  HOME: "/",
  DASHBOARD: "/dashboard",
  JOBS: "/jobs",
  PROFILE: "/profile",
  RESUMES: "/resumes",
  COMMUNICATION: "/communication",
  OUTREACH: "/outreach",
  INTERVIEW: "/interview",
  BULLET_LIBRARY: "/bullet-library",
  FIT_ANALYSIS: "/fit-analysis",
} as const;

// Local storage keys (re-export from storageKeys.ts)
export { STORAGE_KEYS } from "./storageKeys";
