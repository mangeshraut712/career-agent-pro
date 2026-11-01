"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";

/**
 * Multi-Track Outreach Component
 *
 * 3-tier outreach strategy for job applications:
 * - Direct Connection
 * - Warm Introduction
 * - Cold Outreach
 */

interface OutreachTrack {
  id: string;
  name: string;
  description: string;
  priority: number;
  color: string;
  icon: string;
}

const OUTREACH_TRACKS: OutreachTrack[] = [
  {
    id: "direct",
    name: "Direct Connection",
    description: "Apply directly and leverage existing connections",
    priority: 1,
    color: "green",
    icon: "🎯",
  },
  {
    id: "warm",
    name: "Warm Introduction",
    description: "Get introduced through mutual connections",
    priority: 2,
    color: "blue",
    icon: "🤝",
  },
  {
    id: "cold",
    name: "Cold Outreach",
    description: "Reach out directly to hiring managers",
    priority: 3,
    color: "purple",
    icon: "✉️",
  },
];

interface OutreachMessage {
  tier: number;
  subject: string;
  body: string;
  timing: string;
}

export default function MultiTrackOutreach({
  jobTitle,
  company,
  hiringManager,
}: {
  jobTitle?: string;
  company?: string;
  hiringManager?: string;
}) {
  const [selectedTrack, setSelectedTrack] = useState<string>("direct");
  const [selectedTier, setSelectedTier] = useState<number>(1);

  const getTrackColor = (trackId: string) => {
    const colors: Record<string, string> = {
      green: "from-green-500 to-emerald-600",
      blue: "from-blue-500 to-cyan-600",
      purple: "from-purple-500 to-pink-600",
    };
    const track = OUTREACH_TRACKS.find((t) => t.id === trackId);
    return colors[track?.color || "blue"];
  };

  const getMessages = (trackId: string): OutreachMessage[] => {
    const messages: Record<string, OutreachMessage[]> = {
      direct: [
        {
          tier: 1,
          subject: `Application for ${jobTitle || "Product Manager"} Role`,
          body: `Dear Hiring Team,\n\nI'm excited to apply for the ${jobTitle || "Product Manager"} position at ${company || "your company"}. With my background in [YOUR BACKGROUND], I'm confident I can contribute to [COMPANY GOAL].\n\nHighlights:\n• [Key Achievement 1]\n• [Key Achievement 2]\n• [Key Achievement 3]\n\nI'd love to discuss how my experience aligns with your needs.\n\nBest regards,\n[Your Name]`,
          timing: "Within 24 hours of posting",
        },
        {
          tier: 2,
          subject: `Following Up: ${jobTitle || "Product Manager"} Application`,
          body: `Hi [Hiring Manager],\n\nI wanted to follow up on my application for the ${jobTitle || "Product Manager"} role submitted on [DATE].\n\nI'm particularly excited about [SPECIFIC PROJECT/INITIATIVE] and how my experience with [RELEVANT SKILL] could contribute.\n\nWould you be available for a brief call this week?\n\nBest,\n[Your Name]`,
          timing: "3-5 days after application",
        },
        {
          tier: 3,
          subject: `Final Follow-Up: ${jobTitle || "Product Manager"} Opportunity`,
          body: `Hi [Hiring Manager],\n\nI hope this email finds you well. I'm following up once more on the ${jobTitle || "Product Manager"} position.\n\nI understand you're likely evaluating many candidates. If there's any additional information I can provide, please let me know.\n\nI remain very interested and excited about this opportunity.\n\nThank you for your consideration.\n\n[Your Name]`,
          timing: "7-10 days after Tier 2",
        },
      ],
      warm: [
        {
          tier: 1,
          subject: `Introduction to ${company || "Company"} Team`,
          body: `Hi [Mutual Connection],\n\nI hope you're doing well! I'm reaching out because I noticed ${company || "your company"} is hiring for a ${jobTitle || "Product Manager"} role.\n\nGiven my background in [YOUR EXPERIENCE], I think I could be a great fit. Would you be comfortable making an introduction to the hiring team?\n\nI'd be happy to send you my resume and any other info that would be helpful.\n\nThanks for considering!\n\n[Your Name]`,
          timing: "Before applying",
        },
        {
          tier: 2,
          subject: `Thank You + Application Update`,
          body: `Hi [Mutual Connection],\n\nThank you so much for the introduction to [Hiring Manager]!\n\nI wanted to let you know I've submitted my application and had a great conversation about the role.\n\nI really appreciate you taking the time to connect us.\n\nBest,\n[Your Name]`,
          timing: "After introduction",
        },
        {
          tier: 3,
          subject: `Quick Update on ${company || "Company"} Opportunity`,
          body: `Hi [Mutual Connection],\n\nJust wanted to give you a quick update - I've moved forward in the interview process at ${company || "the company"}!\n\nThank you again for the introduction. It made a huge difference.\n\nI'll keep you posted on how it goes!\n\n[Your Name]`,
          timing: "After interview",
        },
      ],
      cold: [
        {
          tier: 1,
          subject: `${jobTitle || "Product Manager"} at ${company || "Your Company"} - Quick Question`,
          body: `Hi ${hiringManager || "[Hiring Manager]"},\n\nI came across the ${jobTitle || "Product Manager"} opening at ${company || "your company"} and was immediately drawn to [SPECIFIC ASPECT].\n\nWith my experience in [YOUR BACKGROUND], I think I could make an immediate impact on [TEAM GOAL].\n\nWould you have 15 minutes this week for a brief chat about the role?\n\nBest,\n[Your Name]\n\nP.S. My background: [1-2 sentence summary]`,
          timing: "As soon as posting goes live",
        },
        {
          tier: 2,
          subject: `Following Up: ${jobTitle || "Product Manager"} Conversation`,
          body: `Hi ${hiringManager || "[Hiring Manager]"},\n\nI wanted to follow up on my message from [DATE] about the ${jobTitle || "Product Manager"} role.\n\nI've been following ${company || "your company's"} [RECENT NEWS/PRODUCT LAUNCH] and it reinforced my excitement about this opportunity.\n\nQuick question: What's the biggest challenge the team is facing right now?\n\nHappy to share how I've tackled similar challenges at [YOUR COMPANY].\n\nBest,\n[Your Name]`,
          timing: "5-7 days after Tier 1",
        },
        {
          tier: 3,
          subject: `Last Check-In: ${jobTitle || "Product Manager"} Role`,
          body: `Hi ${hiringManager || "[Hiring Manager]"},\n\nI know you're busy, so I'll keep this brief.\n\nStill very interested in the ${jobTitle || "Product Manager"} role. If the timing isn't right now, I'd love to stay in touch for future opportunities.\n\nEither way, I wish you and the team continued success!\n\nBest,\n[Your Name]`,
          timing: "10-14 days after Tier 2",
        },
      ],
    };

    return messages[trackId] || messages.direct;
  };

  const currentMessages = getMessages(selectedTrack);
  const currentMessage =
    currentMessages.find((m) => m.tier === selectedTier) || currentMessages[0];

  return (
    <div className="w-full space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          Multi-Track Outreach Strategy
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          3-tier approach for maximum response rate
        </p>
      </div>

      {/* Track Selection */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {OUTREACH_TRACKS.map((track) => (
          <button
            key={track.id}
            onClick={() => setSelectedTrack(track.id)}
            className={`p-6 rounded-xl border-2 transition-all text-left ${
              selectedTrack === track.id
                ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20 shadow-lg"
                : "border-gray-200 dark:border-gray-700 hover:border-gray-300"
            }`}
          >
            <div className="flex items-start justify-between mb-3">
              <span className="text-3xl">{track.icon}</span>
              <span
                className={`text-xs px-2 py-1 rounded-full ${
                  track.priority === 1
                    ? "bg-green-100 text-green-700"
                    : track.priority === 2
                      ? "bg-blue-100 text-blue-700"
                      : "bg-purple-100 text-purple-700"
                }`}
              >
                Priority {track.priority}
              </span>
            </div>
            <h3 className="font-bold text-gray-900 dark:text-white mb-1">
              {track.name}
            </h3>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              {track.description}
            </p>
          </button>
        ))}
      </div>

      {/* Tier Selection */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
          Message Sequence
        </h3>
        <div className="flex gap-3">
          {[1, 2, 3].map((tier) => (
            <button
              key={tier}
              onClick={() => setSelectedTier(tier)}
              className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all ${
                selectedTier === tier
                  ? `bg-gradient-to-r ${getTrackColor(selectedTrack)} text-white shadow-lg`
                  : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400"
              }`}
            >
              Tier {tier}
            </button>
          ))}
        </div>
      </div>

      {/* Message Template */}
      <motion.div
        key={`${selectedTrack}-${selectedTier}`}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-700 shadow-lg p-6"
      >
        {/* Timing Badge */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-gray-900 dark:text-white">
            Message Template
          </h3>
          <span className="text-xs px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full">
            ⏱️ {currentMessage.timing}
          </span>
        </div>

        {/* Subject */}
        <div className="mb-4">
          <label className="block text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2">
            SUBJECT
          </label>
          <input
            type="text"
            value={currentMessage.subject}
            readOnly
            className="w-full px-4 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg font-medium text-gray-900 dark:text-white"
          />
        </div>

        {/* Body */}
        <div>
          <label className="block text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2">
            MESSAGE
          </label>
          <textarea
            value={currentMessage.body}
            readOnly
            rows={12}
            className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm text-gray-900 dark:text-white font-mono"
          />
        </div>

        {/* Copy Button */}
        <button
          onClick={() => {
            navigator.clipboard.writeText(
              `Subject: ${currentMessage.subject}\n\n${currentMessage.body}`,
            );
          }}
          className="mt-4 w-full py-3 px-6 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white rounded-lg font-semibold shadow-lg hover:shadow-xl transition-all"
        >
          📋 Copy to Clipboard
        </button>
      </motion.div>

      {/* Pro Tips */}
      <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-2xl border border-yellow-200 dark:border-yellow-800 p-6">
        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <span>💡</span> Pro Tips
        </h3>
        <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
          <li className="flex items-start gap-2">
            <span className="text-yellow-600">•</span>
            <span>
              Personalize every message with specific company/role details
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-yellow-600">•</span>
            <span>Replace [PLACEHOLDERS] with your actual information</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-yellow-600">•</span>
            <span>
              Send follow-ups on Tuesday-Thursday mid-morning for best response
              rates
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-yellow-600">•</span>
            <span>
              Track all outreach in a spreadsheet for follow-up management
            </span>
          </li>
        </ul>
      </div>
    </div>
  );
}
