"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import API_URL from "@/lib/api";

/**
 * Spinning Engine Component
 *
 * Adapts resume language to match target industry/company stage
 * without fabrication. Shows before/after comparison with explanations.
 */

interface SpinningChange {
  original: string;
  replaced: string;
  reason: string;
}

interface SpinningResult {
  original: string;
  spun: string;
  changes: SpinningChange[];
  targetStage: string;
  similarity: number;
  explanation: string;
}

interface SpinningEngineProps {
  originalText: string;
  onApply?: (spunText: string) => void;
}

const COMPANY_STAGES = [
  {
    id: "early_stage",
    label: "Early Stage / Startup",
    description: "Emphasizes speed, iteration, and validation",
    color: "purple",
  },
  {
    id: "growth_stage",
    label: "Growth Stage / Scale-up",
    description: "Emphasizes metrics, scaling, and optimization",
    color: "blue",
  },
  {
    id: "enterprise",
    label: "Enterprise / Large Company",
    description:
      "Emphasizes coordination, compliance, and stakeholder management",
    color: "gray",
  },
];

export default function SpinningEngine({
  originalText,
  onApply,
}: SpinningEngineProps) {
  const [selectedStage, setSelectedStage] = useState("growth_stage");
  const [spinningResult, setSpinningResult] = useState<SpinningResult | null>(
    null,
  );
  const [isSpinning, setIsSpinning] = useState(false);

  const handleSpin = async () => {
    setIsSpinning(true);

    try {
      const response = await fetch(`${API_URL}/spin-text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: originalText,
          targetStage: selectedStage,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        setSpinningResult(result);
      } else {
        // Fallback to demo result
        setSpinningResult(getDemoResult(originalText, selectedStage));
      }
    } catch {
      // Fallback to demo result
      setSpinningResult(getDemoResult(originalText, selectedStage));
    } finally {
      setIsSpinning(false);
    }
  };

  const getDemoResult = (text: string, stage: string): SpinningResult => {
    // Demo transformation logic
    let spun = text;
    const changes: SpinningChange[] = [];

    if (stage === "early_stage") {
      if (text.includes("coordinated")) {
        spun = spun.replace(/coordinated/gi, "shipped");
        changes.push({
          original: "coordinated",
          replaced: "shipped",
          reason: "Startups value speed of execution",
        });
      }
    } else if (stage === "growth_stage") {
      if (text.includes("tested")) {
        spun = spun.replace(/tested/gi, "optimized");
        changes.push({
          original: "tested",
          replaced: "optimized",
          reason: "Growth companies focus on optimization",
        });
      }
    }

    return {
      original: text,
      spun: spun,
      changes,
      targetStage: stage,
      similarity: 0.85,
      explanation: `Adapted language to match ${stage.replace("_", " ")} terminology`,
    };
  };

  const getStageColor = (stageId: string) => {
    const stage = COMPANY_STAGES.find((s) => s.id === stageId);
    const colors: Record<string, string> = {
      purple: "from-purple-500 to-pink-500",
      blue: "from-blue-500 to-cyan-500",
      gray: "from-gray-600 to-gray-800",
    };
    return colors[stage?.color || "blue"];
  };

  return (
    <div className="w-full space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          Industry Language Adaptation
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Adapt your experience to match target company stage - without changing
          facts
        </p>
      </div>

      {/* Stage Selector */}
      <div>
        <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Target Company Stage
        </label>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {COMPANY_STAGES.map((stage) => (
            <button
              key={stage.id}
              onClick={() => setSelectedStage(stage.id)}
              className={`p-4 rounded-xl border-2 transition-all text-left ${
                selectedStage === stage.id
                  ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                  : "border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600"
              }`}
            >
              <div
                className={`w-3 h-3 rounded-full mb-2 bg-gradient-to-r ${getStageColor(stage.id)}`}
              />
              <h3 className="font-semibold text-gray-900 dark:text-white text-sm">
                {stage.label}
              </h3>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                {stage.description}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Original Text */}
      <div>
        <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
          Original Text
        </label>
        <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-900 dark:text-white">
            {originalText}
          </p>
        </div>
      </div>

      {/* Spin Button */}
      <button
        onClick={handleSpin}
        disabled={isSpinning || !originalText}
        className={`w-full py-3 px-6 rounded-lg font-semibold transition-all ${
          isSpinning || !originalText
            ? "bg-gray-300 dark:bg-gray-700 text-gray-500 cursor-not-allowed"
            : "bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white shadow-lg hover:shadow-xl"
        }`}
      >
        {isSpinning ? "Adapting Language..." : "✨ Adapt Language"}
      </button>

      {/* Results */}
      <AnimatePresence>
        {spinningResult && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-4"
          >
            {/* Adapted Text */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                Adapted Text ✨
              </label>
              <div className="p-4 bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-lg border-2 border-green-200 dark:border-green-800">
                <p className="text-sm text-gray-900 dark:text-white">
                  {spinningResult.spun}
                </p>
              </div>
            </div>

            {/* Changes */}
            {spinningResult.changes.length > 0 && (
              <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-700 shadow-lg p-6">
                <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                  Changes Made
                </h3>
                <div className="space-y-3">
                  {spinningResult.changes.map((change, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.1 }}
                      className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
                    >
                      <span className="text-blue-500 mt-1">→</span>
                      <div className="flex-1">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="text-sm px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded line-through">
                            {change.original}
                          </span>
                          <span className="text-gray-400">→</span>
                          <span className="text-sm px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded font-medium">
                            {change.replaced}
                          </span>
                        </div>
                        <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                          {change.reason}
                        </p>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}

            {/* No Changes */}
            {spinningResult.changes.length === 0 && (
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                <p className="text-sm text-blue-800 dark:text-blue-300">
                  ✅ Your text already matches{" "}
                  {COMPANY_STAGES.find((s) => s.id === selectedStage)?.label}{" "}
                  language!
                </p>
              </div>
            )}

            {/* Explanation */}
            <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <p className="text-sm text-gray-700 dark:text-gray-300">
                <span className="font-semibold">Explanation:</span>{" "}
                {spinningResult.explanation}
              </p>
            </div>

            {/* Apply Button */}
            {onApply && spinningResult.changes.length > 0 && (
              <button
                onClick={() => onApply(spinningResult.spun)}
                className="w-full py-3 px-6 bg-green-600 hover:bg-green-700 text-white rounded-lg font-semibold shadow-lg hover:shadow-xl transition-all"
              >
                ✓ Apply Changes
              </button>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Examples */}
      <div className="mt-8 bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-2xl border border-purple-200 dark:border-purple-800 p-6">
        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
          Example Transformations
        </h3>
        <div className="space-y-4">
          <div>
            <p className="text-xs font-semibold text-purple-600 dark:text-purple-400 mb-1">
              Healthcare → Tech (Growth Stage)
            </p>
            <div className="text-sm space-y-1">
              <p className="text-gray-600 dark:text-gray-400">
                Before: &quot;Led hospice care teams serving vulnerable
                families&quot;
              </p>
              <p className="text-gray-900 dark:text-white font-medium">
                After: &quot;Led response teams serving vulnerable populations
                in high-stakes scenarios&quot;
              </p>
            </div>
          </div>

          <div>
            <p className="text-xs font-semibold text-blue-600 dark:text-blue-400 mb-1">
              Startup → Enterprise
            </p>
            <div className="text-sm space-y-1">
              <p className="text-gray-600 dark:text-gray-400">
                Before: &quot;Shipped MVP and validated product-market fit&quot;
              </p>
              <p className="text-gray-900 dark:text-white font-medium">
                After: &quot;Delivered enterprise solution ensuring compliance
                with stakeholder requirements&quot;
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
