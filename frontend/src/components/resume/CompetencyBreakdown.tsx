"use client";

import React from "react";
import { motion } from "framer-motion";

/**
 * Competency Breakdown Component
 *
 * Displays JD competency analysis with visual charts showing:
 * - Competency weightage (what the JD emphasizes)
 * - Your skill match per competency
 * - Gaps and recommendations
 */

interface Competency {
  name: string;
  weightage: number; // Percentage 0-100
  yourScore?: number; // Your match score 0-100
  keywordsFound?: string[];
  gap?: number;
}

interface CompetencyBreakdownProps {
  competencies: Competency[];
  overallFit?: number;
  companyStage?: string;
  recommendations?: string[];
}

export default function CompetencyBreakdown({
  competencies,
  overallFit,
  companyStage,
  recommendations = [],
}: CompetencyBreakdownProps) {
  const getColorForScore = (score: number): string => {
    if (score >= 80) return "bg-green-500";
    if (score >= 60) return "bg-yellow-500";
    return "bg-red-500";
  };

  const getTextColorForScore = (score: number): string => {
    if (score >= 80) return "text-green-600 dark:text-green-400";
    if (score >= 60) return "text-yellow-600 dark:text-yellow-400";
    return "text-red-600 dark:text-red-400";
  };

  const getStageColor = (stage: string): string => {
    const colors: Record<string, string> = {
      early_stage: "bg-purple-500",
      growth_stage: "bg-blue-500",
      enterprise: "bg-gray-600",
    };
    return colors[stage] || "bg-gray-500";
  };

  const getStageLabel = (stage: string): string => {
    const labels: Record<string, string> = {
      early_stage: "Early Stage / Startup",
      growth_stage: "Growth Stage / Scale-up",
      enterprise: "Enterprise / Large Company",
    };
    return labels[stage] || stage;
  };

  // Sort competencies by weightage
  const sortedByWeightage = [...competencies].sort(
    (a, b) => b.weightage - a.weightage,
  );

  // Get top strengths and gaps (if yourScore is available)
  const topStrengths = [...competencies]
    .filter((c) => c.yourScore !== undefined)
    .sort((a, b) => (b.yourScore || 0) - (a.yourScore || 0))
    .slice(0, 3);

  const topGaps = [...competencies]
    .filter((c) => c.gap !== undefined && c.gap > 0)
    .sort((a, b) => (b.gap || 0) - (a.gap || 0))
    .slice(0, 3);

  return (
    <div className="w-full space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          Competency Breakdown
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Analysis of what this job emphasizes and how you match up
        </p>
      </div>

      {/* Overall Fit Score (if available) */}
      {overallFit !== undefined && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-6 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl text-white shadow-lg"
        >
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold">Overall Fit Score</h3>
              <p className="text-sm opacity-90 mt-1">
                How well you match this role
              </p>
            </div>
            <div className="text-5xl font-bold">
              {overallFit}
              <span className="text-2xl">/100</span>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="mt-4 w-full h-3 bg-white/20 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-white"
              initial={{ width: 0 }}
              animate={{ width: `${overallFit}%` }}
              transition={{ duration: 1, delay: 0.2 }}
            />
          </div>
        </motion.div>
      )}

      {/* Company Stage */}
      {companyStage && (
        <div className="flex items-center gap-3 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <div
            className={`w-3 h-3 rounded-full ${getStageColor(companyStage)}`}
          />
          <div>
            <h4 className="text-sm font-semibold text-gray-900 dark:text-white">
              Company Stage
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {getStageLabel(companyStage)}
            </p>
          </div>
        </div>
      )}

      {/* Job Emphasis (Weightage) */}
      <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-700 shadow-lg p-6">
        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
          What This Job Emphasizes
        </h3>

        <div className="space-y-4">
          {sortedByWeightage.map((comp, index) => (
            <motion.div
              key={comp.name}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {comp.name}
                </span>
                <span className="text-sm font-bold text-blue-600 dark:text-blue-400">
                  {comp.weightage.toFixed(1)}%
                </span>
              </div>

              {/* Weightage Bar */}
              <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-blue-500 to-purple-600"
                  initial={{ width: 0 }}
                  animate={{ width: `${comp.weightage}%` }}
                  transition={{ duration: 0.8, delay: index * 0.1 + 0.2 }}
                />
              </div>

              {/* Keywords */}
              {comp.keywordsFound && comp.keywordsFound.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {comp.keywordsFound.slice(0, 3).map((keyword, i) => (
                    <span
                      key={i}
                      className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full"
                    >
                      {keyword}
                    </span>
                  ))}
                </div>
              )}
            </motion.div>
          ))}
        </div>
      </div>

      {/* Your Match (if scores available) */}
      {competencies.some((c) => c.yourScore !== undefined) && (
        <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-700 shadow-lg p-6">
          <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
            Your Skill Match
          </h3>

          <div className="space-y-4">
            {competencies
              .filter((c) => c.yourScore !== undefined)
              .sort((a, b) => (b.yourScore || 0) - (a.yourScore || 0))
              .map((comp, index) => (
                <motion.div
                  key={comp.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      {comp.name}
                    </span>
                    <span
                      className={`text-sm font-bold ${getTextColorForScore(comp.yourScore || 0)}`}
                    >
                      {comp.yourScore}/100
                    </span>
                  </div>

                  {/* Score Bar */}
                  <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <motion.div
                      className={`h-full ${getColorForScore(comp.yourScore || 0)}`}
                      initial={{ width: 0 }}
                      animate={{ width: `${comp.yourScore}%` }}
                      transition={{ duration: 0.8, delay: index * 0.1 + 0.2 }}
                    />
                  </div>
                </motion.div>
              ))}
          </div>
        </div>
      )}

      {/* Strengths & Gaps */}
      {(topStrengths.length > 0 || topGaps.length > 0) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Top Strengths */}
          {topStrengths.length > 0 && (
            <div className="bg-green-50 dark:bg-green-900/20 rounded-2xl border border-green-200 dark:border-green-800 p-6">
              <h3 className="text-lg font-bold text-green-800 dark:text-green-300 mb-4">
                ✅ Top Strengths
              </h3>
              <div className="space-y-3">
                {topStrengths.map((comp, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <span className="text-sm font-medium text-green-700 dark:text-green-400">
                      {comp.name}
                    </span>
                    <span className="text-sm font-bold text-green-600 dark:text-green-300">
                      {comp.yourScore}/100
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Top Gaps */}
          {topGaps.length > 0 && (
            <div className="bg-red-50 dark:bg-red-900/20 rounded-2xl border border-red-200 dark:border-red-800 p-6">
              <h3 className="text-lg font-bold text-red-800 dark:text-red-300 mb-4">
                ⚠️ Areas to Improve
              </h3>
              <div className="space-y-3">
                {topGaps.map((comp, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <span className="text-sm font-medium text-red-700 dark:text-red-400">
                      {comp.name}
                    </span>
                    <span className="text-sm font-bold text-red-600 dark:text-red-300">
                      {comp.gap} points gap
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-2xl border border-blue-200 dark:border-blue-800 p-6">
          <h3 className="text-lg font-bold text-blue-800 dark:text-blue-300 mb-4">
            💡 Recommendations
          </h3>
          <ul className="space-y-2">
            {recommendations.map((rec, i) => (
              <li
                key={i}
                className="text-sm text-blue-700 dark:text-blue-400 flex items-start gap-2"
              >
                <span className="text-blue-500 mt-0.5">•</span>
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
