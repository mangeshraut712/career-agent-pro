/**
 * Error handling utilities
 */

export interface ApiError {
  message: string;
  status?: number;
  code?: string;
}

/**
 * Extract error message from various error types
 */
export function getErrorMessage(error: unknown): string {
  if (typeof error === "string") return error;

  if (error instanceof Error) return error.message;

  if (error && typeof error === "object") {
    if ("message" in error && typeof error.message === "string") {
      return error.message;
    }
    if ("error" in error && typeof error.error === "string") {
      return error.error;
    }
  }

  return "An unexpected error occurred";
}

/**
 * Handle API errors consistently
 */
export function handleApiError(error: unknown): ApiError {
  const message = getErrorMessage(error);

  // Type guard for error with response
  const status =
    error &&
    typeof error === "object" &&
    "response" in error &&
    error.response &&
    typeof error.response === "object" &&
    "status" in error.response
      ? (error.response.status as number)
      : error && typeof error === "object" && "status" in error
        ? (error.status as number)
        : undefined;

  const code =
    error &&
    typeof error === "object" &&
    "response" in error &&
    error.response &&
    typeof error.response === "object" &&
    "data" in error.response &&
    error.response.data &&
    typeof error.response.data === "object" &&
    "code" in error.response.data
      ? (error.response.data.code as string)
      : error && typeof error === "object" && "code" in error
        ? (error.code as string)
        : undefined;

  return {
    message,
    status,
    code,
  };
}

/**
 * Toast error helper
 */
export function toastError(
  toast: (message: string, type: string) => void,
  error: unknown,
): void {
  const message = getErrorMessage(error);
  toast(message, "error");
}

/**
 * Toast success helper
 */
export function toastSuccess(
  toast: (message: string, type: string) => void,
  message: string,
): void {
  toast(message, "success");
}
