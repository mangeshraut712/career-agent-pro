"use client";

import React, { useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import API_URL from "@/lib/api";

/**
 * Resume Verifier Component
 *
 * Comprehensive quality verification before export.
 * Shows quality score, errors, warnings, and suggestions.
 */

interface QualityReport {
  overall_quality_score: number;
  can_export: boolean;
  overall_pass: boolean;
  profile_validation: {
    score: number;
    errors: string[];
    warnings: string[];
    completeness: {
      percentage: number;
      missing_fields: string[];
    };
  };
  bullet_validation: Array<{
    bullet_index: number;
    is_valid: boolean;
    quality_score: number;
    character_count: number;
    has_metrics: boolean;
    errors: string[];
    warnings: string[];
  }>;
  errors: string[];
  warnings: string[];
  suggestions: string[];
  quality_report: {
    quality_level: string;
    quality_emoji: string;
    summary: string;
    bullet_stats?: {
      total_bullets: number;
      valid_bullets: number;
      avg_quality: number;
      bullets_with_metrics: number;
    };
  };
}

interface ResumeVerifierProps {
  resume: Record<string, unknown>;
  bullets?: Array<Record<string, unknown>>;
  onExport?: () => void;
  strictMode?: boolean;
}

export default function ResumeVerifier({
  resume,
  bullets,
  onExport,
  strictMode = false,
}: ResumeVerifierProps) {
  const [verificationResult, setVerificationResult] =
    React.useState<QualityReport | null>(null);
  const [isVerifying, setIsVerifying] = React.useState(false);
  const [showDetails, setShowDetails] = React.useState(false);

  const verifyResume = useCallback(async () => {
    setIsVerifying(true);

    try {
      const response = await fetch(`${API_URL}/verify-resume-quality`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          resume,
          bullets,
          strict_mode: strictMode,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        setVerificationResult(result);
      }
    } catch (err) {
      console.error("Verification failed:", err);
    } finally {
      setIsVerifying(false);
    }
  }, [resume, bullets, strictMode]);

  React.useEffect(() => {
    if (resume) {
      verifyResume();
    }
  }, [resume, verifyResume]);

  const getScoreColor = (score: number): string => {
    if (score >= 85) return "from-green-500 to-emerald-600";
    if (score >= 70) return "from-yellow-500 to-orange-500";
    return "from-red-500 to-pink-600";
  };

  const getScoreTextColor = (score: number): string => {
    if (score >= 85) return "text-green-600 dark:text-green-400";
    if (score >= 70) return "text-yellow-600 dark:text-yellow-400";
    return "text-red-600 dark:text-red-400";
  };

  if (isVerifying) {
    return (
      <div className="flex items-center justify-center p-12">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-600 dark:text-gray-400">
            Verifying resume quality...
          </p>
        </div>
      </div>
    );
  }

  if (!verificationResult) {
    return (
      <div className="p-8 text-center text-gray-500">
        No verification results yet. Loading...
      </div>
    );
  }

  const {
    quality_report,
    profile_validation,
    bullet_validation,
    errors,
    warnings,
    suggestions,
  } = verificationResult;

  return (
    <div className="w-full space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          Resume Quality Report
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Comprehensive quality verification before export
        </p>
      </div>

      {/* Overall Score Card */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className={`relative overflow-hidden rounded-2xl bg-gradient-to-r ${getScoreColor(verificationResult.overall_quality_score)} p-8 text-white shadow-2xl`}
      >
        <div className="relative z-10">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium opacity-90">
                Overall Quality Score
              </p>
              <h3 className="text-5xl font-bold mt-2">
                {verificationResult.overall_quality_score}
                <span className="text-2xl">/100</span>
              </h3>
              <p className="text-lg mt-2 opacity-90">
                {quality_report.quality_emoji} {quality_report.quality_level}
              </p>
            </div>

            <div className="text-right">
              <div
                className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${
                  verificationResult.can_export ? "bg-white/20" : "bg-black/20"
                }`}
              >
                <span className="text-2xl">
                  {verificationResult.can_export ? "✅" : "🚫"}
                </span>
                <span className="font-semibold">
                  {verificationResult.can_export
                    ? "Ready to Export"
                    : "Cannot Export"}
                </span>
              </div>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="mt-6 w-full h-3 bg-white/20 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-white"
              initial={{ width: 0 }}
              animate={{
                width: `${verificationResult.overall_quality_score}%`,
              }}
              transition={{ duration: 1, delay: 0.2 }}
            />
          </div>

          <p className="mt-4 text-sm opacity-90">{quality_report.summary}</p>
        </div>

        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-10">
          <div
            className="absolute inset-0"
            style={{
              backgroundImage:
                "radial-gradient(circle at 2px 2px, white 1px, transparent 0)",
              backgroundSize: "40px 40px",
            }}
          />
        </div>
      </motion.div>

      {/* Profile & Bullet Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Profile Completeness */}
        <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-700 shadow-lg p-6">
          <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
            Profile Completeness
          </h3>

          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Overall
                </span>
                <span
                  className={`text-sm font-bold ${getScoreTextColor(profile_validation.completeness.percentage)}`}
                >
                  {profile_validation.completeness.percentage}%
                </span>
              </div>
              <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <motion.div
                  className={`h-full ${
                    profile_validation.completeness.percentage >= 80
                      ? "bg-green-500"
                      : profile_validation.completeness.percentage >= 60
                        ? "bg-yellow-500"
                        : "bg-red-500"
                  }`}
                  initial={{ width: 0 }}
                  animate={{
                    width: `${profile_validation.completeness.percentage}%`,
                  }}
                  transition={{ duration: 0.8 }}
                />
              </div>
            </div>

            {profile_validation.completeness.missing_fields.length > 0 && (
              <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                <p className="text-xs font-semibold text-gray-600 dark:text-gray-400 mb-2">
                  Missing Fields:
                </p>
                <div className="flex flex-wrap gap-2">
                  {profile_validation.completeness.missing_fields.map(
                    (field, i) => (
                      <span
                        key={i}
                        className="text-xs px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-full"
                      >
                        {field}
                      </span>
                    ),
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Bullet Statistics */}
        {quality_report.bullet_stats && (
          <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-700 shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
              Bullet Statistics
            </h3>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Total Bullets
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {quality_report.bullet_stats.total_bullets}
                </p>
              </div>

              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Valid Bullets
                </p>
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {quality_report.bullet_stats.valid_bullets}
                </p>
              </div>

              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Avg Quality
                </p>
                <p
                  className={`text-2xl font-bold ${getScoreTextColor(quality_report.bullet_stats.avg_quality)}`}
                >
                  {quality_report.bullet_stats.avg_quality}
                </p>
              </div>

              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  With Metrics
                </p>
                <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {quality_report.bullet_stats.bullets_with_metrics}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Errors */}
      {errors.length > 0 && (
        <div className="bg-red-50 dark:bg-red-900/20 rounded-2xl border-2 border-red-200 dark:border-red-800 p-6">
          <h3 className="text-lg font-bold text-red-800 dark:text-red-300 mb-4 flex items-center gap-2">
            <span>❌</span> Errors ({errors.length})
          </h3>
          <ul className="space-y-2">
            {errors.map((error, i) => (
              <li
                key={i}
                className="text-sm text-red-700 dark:text-red-400 flex items-start gap-2"
              >
                <span className="text-red-500 mt-0.5">•</span>
                <span>{error}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-2xl border border-yellow-200 dark:border-yellow-800 p-6">
          <h3 className="text-lg font-bold text-yellow-800 dark:text-yellow-300 mb-4 flex items-center gap-2">
            <span>⚠️</span> Warnings ({warnings.length})
          </h3>
          <ul className="space-y-2">
            {warnings.map((warning, i) => (
              <li
                key={i}
                className="text-sm text-yellow-700 dark:text-yellow-400 flex items-start gap-2"
              >
                <span className="text-yellow-500 mt-0.5">•</span>
                <span>{warning}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Suggestions */}
      {suggestions.length > 0 && (
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-2xl border border-blue-200 dark:border-blue-800 p-6">
          <h3 className="text-lg font-bold text-blue-800 dark:text-blue-300 mb-4 flex items-center gap-2">
            <span>💡</span> Suggestions
          </h3>
          <ul className="space-y-2">
            {suggestions.map((suggestion, i) => (
              <li
                key={i}
                className="text-sm text-blue-700 dark:text-blue-400 flex items-start gap-2"
              >
                <span className="text-blue-500 mt-0.5">•</span>
                <span>{suggestion}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Bullet Details (Expandable) */}
      {bullet_validation && bullet_validation.length > 0 && (
        <div>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="text-sm font-medium text-blue-600 dark:text-blue-400 hover:underline mb-4"
          >
            {showDetails ? "▼ Hide Bullet Details" : "▶ Show Bullet Details"}
          </button>

          <AnimatePresence>
            {showDetails && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="space-y-3"
              >
                {bullet_validation.map((bullet, i) => (
                  <div
                    key={i}
                    className={`p-4 rounded-lg border ${
                      bullet.is_valid
                        ? "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800"
                        : "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                        Bullet #{i + 1}
                      </span>
                      <span
                        className={`text-xs font-bold px-2 py-1 rounded-full ${
                          bullet.is_valid
                            ? "bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200"
                            : "bg-red-200 dark:bg-red-800 text-red-800 dark:text-red-200"
                        }`}
                      >
                        {bullet.quality_score}/100
                      </span>
                    </div>

                    <div className="text-xs space-y-1">
                      <p
                        className={
                          bullet.character_count >= 240 &&
                          bullet.character_count <= 260
                            ? "text-green-600"
                            : "text-red-600"
                        }
                      >
                        Characters: {bullet.character_count} / 240-260
                      </p>
                      <p
                        className={
                          bullet.has_metrics ? "text-green-600" : "text-red-600"
                        }
                      >
                        Metrics: {bullet.has_metrics ? "✓ Yes" : "✗ No"}
                      </p>
                    </div>
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Export Button */}
      {onExport && (
        <button
          onClick={onExport}
          disabled={!verificationResult.can_export}
          className={`w-full py-4 px-6 rounded-xl font-bold text-lg transition-all ${
            verificationResult.can_export
              ? "bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white shadow-lg hover:shadow-xl"
              : "bg-gray-300 dark:bg-gray-700 text-gray-500 dark:text-gray-400 cursor-not-allowed"
          }`}
        >
          {verificationResult.can_export
            ? "✓ Export Resume"
            : "🚫 Fix Errors Before Export"}
        </button>
      )}
    </div>
  );
}
