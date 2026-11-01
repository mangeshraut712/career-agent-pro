"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import API_URL from "@/lib/api";

/**
 * 6-Point Bullet Builder Component
 *
 * Professional bullet point creation following the 6-point framework.
 * Features real-time validation, character counting, and AI suggestions.
 */

interface SixPointBullet {
  action: string;
  context: string;
  method: string;
  result: string;
  impact: string;
  outcome: string;
  competency?: string;
  companyStage?: string;
  tags?: string[];
}

interface BulletBuilderProps {
  initialBullet?: Partial<SixPointBullet>;
  onSave?: (bullet: SixPointBullet) => void;
  onCancel?: () => void;
}

interface ValidationResult {
  isValid: boolean;
  characterCount: number;
  hasMetrics: boolean;
  hasAllSixPoints: boolean;
  qualityScore: number;
  errors: string[];
  warnings: string[];
  suggestions: string[];
}

export default function BulletBuilder({
  initialBullet,
  onSave,
  onCancel,
}: BulletBuilderProps) {
  const [bullet, setBullet] = useState<SixPointBullet>({
    action: initialBullet?.action || "",
    context: initialBullet?.context || "",
    method: initialBullet?.method || "",
    result: initialBullet?.result || "",
    impact: initialBullet?.impact || "",
    outcome: initialBullet?.outcome || "",
    competency: initialBullet?.competency || "",
    companyStage: initialBullet?.companyStage || "growth_stage",
    tags: initialBullet?.tags || [],
  });

  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [assembledBullet, setAssembledBullet] = useState("");
  const [showPreview, setShowPreview] = useState(false);

  // Assemble bullet from 6 parts
  useEffect(() => {
    const assembled = assembleBullet(bullet);
    setAssembledBullet(assembled);

    // Auto-validate after user stops typing (debounce)
    const timer = setTimeout(() => {
      validateBullet(assembled);
    }, 500);

    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bullet]);

  const assembleBullet = (b: SixPointBullet): string => {
    let text = `${b.action} ${b.context}`;

    if (b.method) text += `, ${b.method}`;
    if (b.result) text += `, ${b.result}`;
    if (b.impact) text += `, ${b.impact}`;
    if (b.outcome) text += ` ${b.outcome}`;

    return text.replace(/\s+/g, " ").trim();
  };

  const validateBullet = async (bulletText: string) => {
    try {
      // Simulate backend validation (replace with actual API call)
      const response = await fetch(`${API_URL}/validate-bullet`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bullet }),
      });

      if (response.ok) {
        const result = await response.json();
        setValidation(result);
      } else {
        // Fallback to client-side validation
        setValidation(clientSideValidation(bulletText));
      }
    } catch (err) {
      console.error("Validation error:", err);
      // Fallback to client-side validation
      setValidation(clientSideValidation(bulletText));
    }
  };

  const clientSideValidation = (text: string): ValidationResult => {
    const charCount = text.length;
    const hasMetrics = /\d+(?:,\d{3})*(?:\.\d+)?[%$KMB]?|:\d+|\d+x/i.test(text);
    const hasAllSix = Object.values(bullet).every(
      (v) => v && v.toString().trim().length > 0,
    );

    const errors: string[] = [];
    const warnings: string[] = [];
    const suggestions: string[] = [];

    if (charCount < 240)
      errors.push(`Too short (${charCount} chars). Need at least 240.`);
    if (charCount > 260)
      errors.push(`Too long (${charCount} chars). Must be under 260.`);
    if (!hasMetrics) errors.push("Must contain metrics (numbers, %, $)");
    if (!hasAllSix) errors.push("All 6 fields are required");

    const qualityScore = Math.min(
      100,
      Math.max(
        0,
        (hasAllSix ? 30 : 0) +
          (hasMetrics ? 25 : 0) +
          (charCount >= 240 && charCount <= 260 ? 35 : 0) +
          10,
      ),
    );

    return {
      isValid: errors.length === 0,
      characterCount: charCount,
      hasMetrics,
      hasAllSixPoints: hasAllSix,
      qualityScore,
      errors,
      warnings,
      suggestions,
    };
  };

  const handleSave = () => {
    if (validation?.isValid && onSave) {
      onSave(bullet);
    }
  };

  const getCharCountColor = (count: number): string => {
    if (count < 240) return "text-red-500";
    if (count > 260) return "text-red-500";
    if (count >= 240 && count <= 250) return "text-green-500";
    if (count >= 251 && count <= 260) return "text-yellow-500";
    return "text-gray-500";
  };

  const getProgressPercentage = (): number => {
    const count = assembledBullet.length;
    if (count < 240) return (count / 240) * 100;
    if (count > 260) return 100;
    return 100;
  };

  return (
    <div className="w-full max-w-4xl mx-auto bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-700 shadow-lg p-8">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          6-Point Bullet Builder
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Create high-quality resume bullets following the proven 6-point
          framework
        </p>
      </div>

      {/* Character Counter */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Character Count
          </span>
          <span
            className={`text-sm font-bold ${getCharCountColor(assembledBullet.length)}`}
          >
            {assembledBullet.length} / 240-260
          </span>
        </div>

        {/* Progress Bar */}
        <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <motion.div
            className={`h-full ${
              assembledBullet.length >= 240 && assembledBullet.length <= 260
                ? "bg-green-500"
                : assembledBullet.length > 260
                  ? "bg-red-500"
                  : "bg-blue-500"
            }`}
            initial={{ width: 0 }}
            animate={{ width: `${getProgressPercentage()}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>

        {/* Range Markers */}
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0</span>
          <span className="text-green-600 dark:text-green-400">240</span>
          <span className="text-green-600 dark:text-green-400">260</span>
        </div>
      </div>

      {/* 6-Point Form */}
      <div className="space-y-4">
        {/* 1. ACTION */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            1. Action <span className="text-red-500">*</span>
            <span className="text-xs font-normal text-gray-500 ml-2">
              (Strong verb: Led, Built, Designed, etc.)
            </span>
          </label>
          <input
            type="text"
            value={bullet.action}
            onChange={(e) => setBullet({ ...bullet, action: e.target.value })}
            placeholder="Led"
            className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
          />
        </div>

        {/* 2. CONTEXT */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            2. Context <span className="text-red-500">*</span>
            <span className="text-xs font-normal text-gray-500 ml-2">
              (Where/what/who)
            </span>
          </label>
          <input
            type="text"
            value={bullet.context}
            onChange={(e) => setBullet({ ...bullet, context: e.target.value })}
            placeholder="cross-functional team for payment reconciliation platform"
            className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
          />
        </div>

        {/* 3. METHOD */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            3. Method <span className="text-red-500">*</span>
            <span className="text-xs font-normal text-gray-500 ml-2">
              (How you did it)
            </span>
          </label>
          <input
            type="text"
            value={bullet.method}
            onChange={(e) => setBullet({ ...bullet, method: e.target.value })}
            placeholder="using Agile methodology and stakeholder interviews"
            className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
          />
        </div>

        {/* 4. RESULT */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            4. Result <span className="text-red-500">*</span>
            <span className="text-xs font-normal text-gray-500 ml-2">
              (Quantified outcome with metrics!)
            </span>
          </label>
          <input
            type="text"
            value={bullet.result}
            onChange={(e) => setBullet({ ...bullet, result: e.target.value })}
            placeholder="reducing processing time by 40%"
            className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
          />
          {!validation?.hasMetrics && bullet.result && (
            <p className="text-xs text-red-500 mt-1">
              ⚠️ Add metrics (numbers, %, $) to your result
            </p>
          )}
        </div>

        {/* 5. IMPACT */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            5. Impact <span className="text-red-500">*</span>
            <span className="text-xs font-normal text-gray-500 ml-2">
              (Business effect)
            </span>
          </label>
          <input
            type="text"
            value={bullet.impact}
            onChange={(e) => setBullet({ ...bullet, impact: e.target.value })}
            placeholder="improving cash flow visibility"
            className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
          />
        </div>

        {/* 6. OUTCOME */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
            6. Outcome <span className="text-red-500">*</span>
            <span className="text-xs font-normal text-gray-500 ml-2">
              (Strategic value)
            </span>
          </label>
          <input
            type="text"
            value={bullet.outcome}
            onChange={(e) => setBullet({ ...bullet, outcome: e.target.value })}
            placeholder="for Fortune 500 clients"
            className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
          />
        </div>
      </div>

      {/* Preview */}
      <div className="mt-6">
        <button
          onClick={() => setShowPreview(!showPreview)}
          className="text-sm font-medium text-blue-600 dark:text-blue-400 hover:underline"
        >
          {showPreview ? "▼ Hide Preview" : "▶ Show Preview"}
        </button>

        <AnimatePresence>
          {showPreview && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-3 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800"
            >
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                {assembledBullet || "Complete all fields to see preview..."}
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Validation Status */}
      {validation && (
        <div className="mt-6 space-y-3">
          {/* Quality Score */}
          <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Quality Score
            </span>
            <span
              className={`text-2xl font-bold ${
                validation.qualityScore >= 85
                  ? "text-green-500"
                  : validation.qualityScore >= 70
                    ? "text-yellow-500"
                    : "text-red-500"
              }`}
            >
              {validation.qualityScore}/100
            </span>
          </div>

          {/* Errors */}
          {validation.errors.length > 0 && (
            <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
              <h4 className="text-sm font-semibold text-red-800 dark:text-red-300 mb-2">
                ❌ Errors
              </h4>
              <ul className="space-y-1">
                {validation.errors.map((error, i) => (
                  <li
                    key={i}
                    className="text-sm text-red-700 dark:text-red-400"
                  >
                    • {error}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Success */}
          {validation.isValid && (
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
              <p className="text-sm font-medium text-green-800 dark:text-green-300">
                ✅ Perfect! This bullet meets all quality standards.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="mt-8 flex gap-3">
        <button
          onClick={handleSave}
          disabled={!validation?.isValid}
          className={`flex-1 py-3 px-6 rounded-lg font-semibold transition-all ${
            validation?.isValid
              ? "bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:shadow-xl"
              : "bg-gray-300 dark:bg-gray-700 text-gray-500 dark:text-gray-400 cursor-not-allowed"
          }`}
        >
          Save Bullet
        </button>

        {onCancel && (
          <button
            onClick={onCancel}
            className="px-6 py-3 rounded-lg font-semibold text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800 transition-all"
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}
