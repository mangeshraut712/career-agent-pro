"use client";

import React from "react";

interface LoadingSkeletonProps {
  variant?: "text" | "card" | "list" | "table" | "bullet";
  count?: number;
  className?: string;
}

export function LoadingSkeleton({
  variant = "text",
  count = 1,
  className = "",
}: LoadingSkeletonProps) {
  const skeletons = Array.from({ length: count }, (_, i) => i);

  if (variant === "text") {
    return (
      <div className={`space-y-3 ${className}`}>
        {skeletons.map((i) => (
          <div key={i} className="animate-pulse">
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-full"></div>
          </div>
        ))}
      </div>
    );
  }

  if (variant === "card") {
    return (
      <div className={`space-y-4 ${className}`}>
        {skeletons.map((i) => (
          <div
            key={i}
            className="animate-pulse bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
          >
            <div className="flex items-center space-x-4 mb-4">
              <div className="h-12 w-12 bg-gray-200 dark:bg-gray-700 rounded-full"></div>
              <div className="flex-1 space-y-2">
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
                <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
              </div>
            </div>
            <div className="space-y-2">
              <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-full"></div>
              <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-5/6"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (variant === "list") {
    return (
      <div className={`space-y-3 ${className}`}>
        {skeletons.map((i) => (
          <div key={i} className="animate-pulse flex items-center space-x-4">
            <div className="h-10 w-10 bg-gray-200 dark:bg-gray-700 rounded-lg"></div>
            <div className="flex-1 space-y-2">
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
              <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (variant === "table") {
    return (
      <div className={`space-y-2 ${className}`}>
        {/* Header */}
        <div className="animate-pulse grid grid-cols-4 gap-4 pb-3 border-b border-gray-200 dark:border-gray-700">
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="h-4 bg-gray-200 dark:bg-gray-700 rounded"
            ></div>
          ))}
        </div>
        {/* Rows */}
        {skeletons.map((i) => (
          <div key={i} className="animate-pulse grid grid-cols-4 gap-4 py-3">
            {[1, 2, 3, 4].map((j) => (
              <div
                key={j}
                className="h-4 bg-gray-200 dark:bg-gray-700 rounded"
              ></div>
            ))}
          </div>
        ))}
      </div>
    );
  }

  if (variant === "bullet") {
    return (
      <div className={`space-y-4 ${className}`}>
        {skeletons.map((i) => (
          <div
            key={i}
            className="animate-pulse bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700"
          >
            <div className="flex items-start space-x-3">
              <div className="h-2 w-2 bg-gray-300 dark:bg-gray-600 rounded-full mt-2"></div>
              <div className="flex-1 space-y-2">
                <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-full"></div>
                <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-11/12"></div>
                <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-4/5"></div>
              </div>
            </div>
            <div className="mt-3 flex space-x-2">
              <div className="h-6 w-16 bg-gray-200 dark:bg-gray-700 rounded-full"></div>
              <div className="h-6 w-20 bg-gray-200 dark:bg-gray-700 rounded-full"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  return null;
}

// Specific loading components for common use cases
export function DashboardSkeleton() {
  return (
    <div className="space-y-6">
      <div className="animate-pulse">
        <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-1/4 mb-4"></div>
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/3"></div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="animate-pulse bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
          >
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-2/3 mb-4"></div>
            <div className="h-10 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
          </div>
        ))}
      </div>
      <LoadingSkeleton variant="card" count={3} />
    </div>
  );
}

export function BulletLibrarySkeleton() {
  return (
    <div className="space-y-6">
      <div className="animate-pulse bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-1/4 mb-4"></div>
        <div className="h-32 bg-gray-200 dark:bg-gray-700 rounded mb-4"></div>
        <div className="h-10 bg-gray-200 dark:bg-gray-700 rounded w-1/3"></div>
      </div>
      <LoadingSkeleton variant="bullet" count={5} />
    </div>
  );
}

export function JobAnalysisSkeleton() {
  return (
    <div className="space-y-6">
      <LoadingSkeleton variant="card" count={1} />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <LoadingSkeleton variant="card" count={2} />
      </div>
      <LoadingSkeleton variant="list" count={4} />
    </div>
  );
}
