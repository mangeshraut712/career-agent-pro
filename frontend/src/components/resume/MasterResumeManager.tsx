"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

/**
 * Master Resume Manager
 *
 * Manage master resume with reusable bullets.
 * Track which bullets are used for which applications.
 */

interface MasterBullet {
  id: string;
  text: string;
  competency: string;
  companyStage: string;
  qualityScore: number;
  usageCount: number;
  lastUsed?: string;
  usedIn: string[];
}

export default function MasterResumeManager() {
  const [bullets] = useState<MasterBullet[]>([
    {
      id: "1",
      text: "Led cross-functional team for payment reconciliation platform, using Agile methodology, reducing processing time by 40%, improving cash flow visibility for Fortune 500 clients",
      competency: "Product Strategy",
      companyStage: "growth_stage",
      qualityScore: 95,
      usageCount: 3,
      lastUsed: "2024-01-05",
      usedIn: ["Google PM", "Meta PM", "Amazon PM"],
    },
  ]);

  const [selectedBullets, setSelectedBullets] = useState<Set<string>>(
    new Set(),
  );
  const [filterCompetency, setFilterCompetency] = useState<string>("all");
  const [filterStage, setFilterStage] = useState<string>("all");
  const [sortBy, setSortBy] = useState<"quality" | "usage">("quality");

  const competencies = [
    "Product Strategy",
    "Technical",
    "Leadership",
    "Data & Analytics",
    "Execution",
  ];
  const stages = ["early_stage", "growth_stage", "enterprise"];

  const getStageLabel = (stage: string) => {
    const labels: Record<string, string> = {
      early_stage: "Early Stage",
      growth_stage: "Growth",
      enterprise: "Enterprise",
    };
    return labels[stage] || stage;
  };

  const getQualityColor = (score: number) => {
    if (score >= 90) return "text-green-600 dark:text-green-400";
    if (score >= 75) return "text-yellow-600 dark:text-yellow-400";
    return "text-red-600 dark:text-red-400";
  };

  const filteredBullets = bullets
    .filter(
      (b) => filterCompetency === "all" || b.competency === filterCompetency,
    )
    .filter((b) => filterStage === "all" || b.companyStage === filterStage)
    .sort((a, b) => {
      if (sortBy === "quality") return b.qualityScore - a.qualityScore;
      return b.usageCount - a.usageCount;
    });

  const toggleBulletSelection = (id: string) => {
    const newSelected = new Set(selectedBullets);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedBullets(newSelected);
  };

  return (
    <div className="w-full space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          Master Resume Library
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Manage your reusable bullets and track usage
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Total Bullets
          </p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {bullets.length}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Avg Quality
          </p>
          <p className="text-2xl font-bold text-green-600">
            {Math.round(
              bullets.reduce((sum, b) => sum + b.qualityScore, 0) /
                bullets.length,
            )}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
          <p className="text-xs text-gray-500 dark:text-gray-400">Selected</p>
          <p className="text-2xl font-bold text-blue-600">
            {selectedBullets.size}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
          <p className="text-xs text-gray-500 dark:text-gray-400">Total Uses</p>
          <p className="text-2xl font-bold text-purple-600">
            {bullets.reduce((sum, b) => sum + b.usageCount, 0)}
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Filter by Competency
            </label>
            <select
              value={filterCompetency}
              onChange={(e) => setFilterCompetency(e.target.value)}
              className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm"
            >
              <option value="all">All Competencies</option>
              {competencies.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Filter by Stage
            </label>
            <select
              value={filterStage}
              onChange={(e) => setFilterStage(e.target.value)}
              className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm"
            >
              <option value="all">All Stages</option>
              {stages.map((s) => (
                <option key={s} value={s}>
                  {getStageLabel(s)}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Sort By
            </label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as "quality" | "usage")}
              className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm"
            >
              <option value="quality">Quality Score</option>
              <option value="usage">Usage Count</option>
            </select>
          </div>
        </div>
      </div>

      {/* Bullets List */}
      <div className="space-y-3">
        <AnimatePresence>
          {filteredBullets.map((bullet, index) => (
            <motion.div
              key={bullet.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ delay: index * 0.05 }}
              className={`bg-white dark:bg-gray-900 rounded-xl border-2 transition-all ${
                selectedBullets.has(bullet.id)
                  ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                  : "border-gray-200 dark:border-gray-700 hover:border-gray-300"
              }`}
            >
              <div className="p-4">
                {/* Header */}
                <div className="flex items-start justify-between mb-3">
                  <button
                    onClick={() => toggleBulletSelection(bullet.id)}
                    className="flex-1 text-left"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <div
                        className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-all ${
                          selectedBullets.has(bullet.id)
                            ? "bg-blue-600 border-blue-600"
                            : "border-gray-300 dark:border-gray-600"
                        }`}
                      >
                        {selectedBullets.has(bullet.id) && (
                          <svg
                            className="w-3 h-3 text-white"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={3}
                              d="M5 13l4 4L19 7"
                            />
                          </svg>
                        )}
                      </div>
                      <span className="text-xs px-2 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-full">
                        {bullet.competency}
                      </span>
                      <span className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-full">
                        {getStageLabel(bullet.companyStage)}
                      </span>
                    </div>
                    <p className="text-sm text-gray-900 dark:text-white leading-relaxed">
                      {bullet.text}
                    </p>
                  </button>
                </div>

                {/* Metadata */}
                <div className="flex items-center justify-between pt-3 border-t border-gray-200 dark:border-gray-700">
                  <div className="flex items-center gap-4 text-xs">
                    <span
                      className={`font-bold ${getQualityColor(bullet.qualityScore)}`}
                    >
                      Quality: {bullet.qualityScore}/100
                    </span>
                    <span className="text-gray-600 dark:text-gray-400">
                      Used: {bullet.usageCount}x
                    </span>
                    {bullet.lastUsed && (
                      <span className="text-gray-500 dark:text-gray-500">
                        Last: {bullet.lastUsed}
                      </span>
                    )}
                  </div>

                  <div className="flex gap-2">
                    <button className="text-xs px-3 py-1 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded">
                      Edit
                    </button>
                    <button className="text-xs px-3 py-1 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded">
                      Delete
                    </button>
                  </div>
                </div>

                {/* Usage History */}
                {bullet.usedIn.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                    <p className="text-xs font-semibold text-gray-600 dark:text-gray-400 mb-2">
                      Used in applications:
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {bullet.usedIn.map((app, i) => (
                        <span
                          key={i}
                          className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded"
                        >
                          {app}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Actions */}
      {selectedBullets.size > 0 && (
        <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 z-50">
          <motion.div
            initial={{ y: 100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-full shadow-2xl px-6 py-4 flex items-center gap-4"
          >
            <span className="font-semibold">
              {selectedBullets.size} bullet
              {selectedBullets.size !== 1 ? "s" : ""} selected
            </span>
            <button className="px-4 py-2 bg-white/20 hover:bg-white/30 rounded-full font-semibold transition-all">
              Add to Resume
            </button>
            <button className="px-4 py-2 bg-white/20 hover:bg-white/30 rounded-full font-semibold transition-all">
              Export
            </button>
          </motion.div>
        </div>
      )}

      {/* Empty State */}
      {filteredBullets.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500 dark:text-gray-400">
            No bullets match your filters
          </p>
        </div>
      )}
    </div>
  );
}
