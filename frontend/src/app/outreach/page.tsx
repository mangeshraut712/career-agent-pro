"use client";

import { FADE_IN } from "@/lib/animations";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Mail,
  Linkedin,
  MessageSquare,
  Copy,
  RefreshCw,
  Clock,
  Target,
  Users,
  Clipboard,
  Building2,
  Sparkles,
  Layout,
  ShieldCheck,
  Zap,
  Rocket,
  History,
  MousePointer2,
  Check,
  PenTool,
} from "lucide-react";
import { AppleCard } from "@/components/ui/AppleCard";
import { AppleButton } from "@/components/ui/AppleButton";
import { useToast } from "@/components/ui/Toast";
import axios from "axios";
import API_URL from "@/lib/api";

import { useAppData } from "@/hooks/useAppData";
import Link from "next/link";

interface OutreachMessage {
  track: string;
  tier: string;
  channel: string;
  subject: string | null;
  message: string;
  timing: string;
  tips: string[];
}

interface OutreachStrategy {
  job_title: string;
  company: string;
  primary_track: string;
  target_contacts: string[];
  preparation_checklist: string[];
  messages: OutreachMessage[];
}

import type { LucideIcon } from "lucide-react";

const trackIcons: Record<string, LucideIcon> = {
  direct: Send,
  warm_intro: Users,
  cold: Target,
};

const channelIcons: Record<string, LucideIcon> = {
  linkedin: Linkedin,
  email: Mail,
};

const tierGradients: Record<string, string> = {
  tier_1: "from-emerald-500 to-teal-600",
  tier_2: "from-amber-500 to-orange-600",
  tier_3: "from-rose-500 to-pink-600",
};

export default function OutreachStrategyPage() {
  const { toast } = useToast();
  const [isGenerating, setIsGenerating] = useState(false);
  const { currentJob, profile } = useAppData();
  const [strategy, setStrategy] = useState<OutreachStrategy | null>(null);
  const [selectedMessage, setSelectedMessage] =
    useState<OutreachMessage | null>(null);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const generateStrategy = async () => {
    if (!currentJob) {
      toast("Select an objective first", "error");
      return;
    }

    setIsGenerating(true);
    setStrategy(null);
    setSelectedMessage(null);

    try {
      const response = await axios.post(
        `${API_URL}/generate-outreach-strategy`,
        {
          job_data: {
            title: currentJob.title,
            company: currentJob.company,
            description: currentJob.description,
          },
          resume_data: {
            name: profile?.name || "",
            email: profile?.email || "",
            summary: profile?.summary || "",
            skills: profile?.skills || [],
            experience: profile?.experience || [],
            linkedin: profile?.linkedin || "",
          },
        },
      );
      setStrategy(response.data);
      toast("Engagement Grid Online!", "success");
    } catch {
      toast("AI unavailable. Engaging local tactics.", "info");
      const localStrategy: OutreachStrategy = {
        job_title: currentJob.title,
        company: currentJob.company,
        primary_track: "direct",
        target_contacts: ["Hiring Manager", "Dept. VP", "Senior Recruiter"],
        preparation_checklist: [
          "Mirror company tone",
          "Research recent funding/news",
          "Verify contact LinkedIn",
        ],
        messages: [
          {
            track: "direct",
            tier: "tier_1",
            channel: "linkedin",
            subject: null,
            message: `Hi there! 👋\n\nI just applied for the ${currentJob.title} role and wanted to reach out directly...`,
            timing: "Immediate after apply",
            tips: ["Keep it under 100 words", "Tue-Thu 10am best"],
          },
          {
            track: "direct",
            tier: "tier_2",
            channel: "email",
            subject: `Follow up: ${currentJob.title}`,
            message: `Hi, checking in on my ${currentJob.title} application...`,
            timing: "T+4 days",
            tips: ["Cite 1 specific project"],
          },
        ],
      };
      setStrategy(localStrategy);
    } finally {
      setIsGenerating(false);
    }
  };

  const copyMessage = async (message: OutreachMessage, index: number) => {
    const fullMessage = message.subject
      ? `Subject: ${message.subject}\n\n${message.message}`
      : message.message;
    await navigator.clipboard.writeText(fullMessage);
    setCopiedIndex(index);
    toast("Protocol Copied.", "success");
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background via-background to-secondary/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-12 lg:py-20">
        {/* Immersive Header */}
        <motion.div {...FADE_IN} className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-2xl bg-cyan-500/10 flex items-center justify-center text-cyan-500 shadow-lg shadow-cyan-500/5">
              <Target size={24} />
            </div>
            <h2 className="text-sm font-black uppercase tracking-[0.2em] text-cyan-500/80">
              Engagement Command
            </h2>
          </div>
          <h1 className="text-5xl sm:text-6xl font-bold tracking-tight mb-6">
            Outreach <span className="text-cyan-500">Pulse.</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl leading-relaxed">
            Multi-channel networking matrix designed to maximize conversion from
            application to interaction.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-12 gap-10">
          {/* Operation Sidebar */}
          <div className="lg:col-span-4 space-y-8">
            {/* Target Node */}
            <motion.div {...FADE_IN} transition={{ delay: 0.1 }}>
              <AppleCard className="p-8 border-border/40 bg-card/60 backdrop-blur-sm rounded-[2.5rem] shadow-xl">
                <div className="flex items-center justify-between mb-8">
                  <h3 className="text-xs font-black uppercase tracking-widest text-muted-foreground">
                    Target Node
                  </h3>
                  <Link
                    href="/jobs"
                    className="text-[10px] font-black tracking-widest text-primary uppercase hover:underline"
                  >
                    RE-CALIBRATE
                  </Link>
                </div>
                {currentJob ? (
                  <div className="space-y-6">
                    <div className="flex items-center gap-4">
                      <div className="w-16 h-16 rounded-2xl bg-cyan-500/10 flex items-center justify-center text-cyan-500 font-black text-2xl shadow-inner uppercase">
                        {currentJob.company.charAt(0)}
                      </div>
                      <div>
                        <h4 className="font-black text-lg">
                          {currentJob.title}
                        </h4>
                        <p className="text-sm font-bold text-muted-foreground truncate">
                          {currentJob.company}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 px-4 py-2 bg-emerald-500/10 rounded-xl border border-emerald-500/10 text-emerald-600">
                      <ShieldCheck size={14} />
                      <span className="text-[10px] font-black uppercase tracking-widest">
                        Priority Target
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-10 opacity-30">
                    <div className="w-12 h-12 rounded-full border-2 border-dashed border-border flex items-center justify-center mx-auto mb-4">
                      <Building2 size={24} />
                    </div>
                    <p className="text-xs font-black uppercase tracking-widest">
                      Connect Job Objective
                    </p>
                  </div>
                )}
              </AppleCard>
            </motion.div>

            <motion.div {...FADE_IN} transition={{ delay: 0.2 }}>
              <AppleButton
                variant="primary"
                onClick={generateStrategy}
                disabled={isGenerating || !currentJob}
                className="w-full h-16 rounded-[2rem] bg-gradient-to-r from-cyan-500 to-blue-600 shadow-2xl shadow-cyan-500/20 font-black tracking-widest uppercase text-xs gap-3"
              >
                {isGenerating ? (
                  <RefreshCw size={18} className="animate-spin" />
                ) : (
                  <Zap size={18} className="fill-white" />
                )}
                {isGenerating
                  ? "Synthesizing Strategy..."
                  : "Initiate Engagement Matrix"}
              </AppleButton>
            </motion.div>

            {/* Checklist Section */}
            {strategy && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="space-y-6"
              >
                <AppleCard className="p-8 border-border/40 rounded-[2.5rem]">
                  <h3 className="text-sm font-black uppercase tracking-widest mb-6 flex items-center gap-2">
                    <Clipboard size={16} className="text-cyan-500" />
                    Pre-Launch Prep
                  </h3>
                  <div className="space-y-4">
                    {strategy.preparation_checklist.map((item, i) => (
                      <div key={i} className="flex items-start gap-3">
                        <div className="w-5 h-5 rounded-md bg-secondary flex items-center justify-center shrink-0 mt-0.5">
                          <Check size={12} className="text-cyan-500" />
                        </div>
                        <span className="text-xs font-bold text-muted-foreground leading-relaxed">
                          {item}
                        </span>
                      </div>
                    ))}
                  </div>
                </AppleCard>
              </motion.div>
            )}
          </div>

          {/* Execution Canvas */}
          <div className="lg:col-span-8">
            <AnimatePresence mode="wait">
              {strategy ? (
                <motion.div
                  key="strategy"
                  initial={{ opacity: 0, y: 40 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="grid md:grid-cols-2 gap-8"
                >
                  {/* Escalation Sequence */}
                  <div className="space-y-6">
                    <div className="flex items-center gap-2 mb-2">
                      <Layout size={18} className="text-primary" />
                      <h3 className="text-sm font-black uppercase tracking-[0.2em] text-muted-foreground">
                        Escalation Grid
                      </h3>
                    </div>

                    {strategy.messages.map((msg, i) => {
                      // eslint-disable-next-line @typescript-eslint/no-unused-vars
                      const _TrackIcon = trackIcons[msg.track] || Send;
                      const ChannelIcon =
                        channelIcons[msg.channel] || MessageSquare;
                      const isActive = selectedMessage === msg;

                      return (
                        <motion.button
                          key={i}
                          whileHover={{ x: 8, scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          onClick={() => setSelectedMessage(msg)}
                          className={`w-full group p-6 rounded-[2rem] border transition-all text-left relative overflow-hidden ${
                            isActive
                              ? "bg-card border-primary shadow-2xl shadow-primary/5"
                              : "bg-card/40 border-border/40 hover:border-primary/40"
                          }`}
                        >
                          <div
                            className={`absolute top-0 left-0 w-1.5 h-full bg-gradient-to-b ${tierGradients[msg.tier] || "from-primary to-blue-500"}`}
                          />
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <div
                                className={`w-8 h-8 rounded-lg ${isActive ? "bg-primary text-white" : "bg-secondary text-primary"} flex items-center justify-center`}
                              >
                                <ChannelIcon size={14} />
                              </div>
                              <span className="text-xs font-black uppercase tracking-widest">
                                {msg.channel} protocol
                              </span>
                            </div>
                            <span
                              className={`text-[10px] font-black uppercase tracking-widest px-2 py-1 rounded-md bg-secondary group-hover:bg-primary/10 transition-colors`}
                            >
                              {msg.tier.replace("_", " ")}
                            </span>
                          </div>
                          <h4 className="text-sm font-bold truncate mb-2">
                            {msg.subject || "Direct Connection Request"}
                          </h4>
                          <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-widest opacity-40">
                            <Clock size={12} />
                            {msg.timing}
                          </div>
                        </motion.button>
                      );
                    })}
                  </div>

                  {/* Message Display Area */}
                  <div className="space-y-6">
                    <div className="flex items-center gap-2 mb-2">
                      <PenTool size={18} className="text-primary" />
                      <h3 className="text-sm font-black uppercase tracking-[0.2em] text-muted-foreground">
                        Tactical Output
                      </h3>
                    </div>

                    <AppleCard className="p-8 border-border/40 bg-card/60 backdrop-blur-xl rounded-[3rem] shadow-2xl min-h-[500px] flex flex-col">
                      {selectedMessage ? (
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="flex flex-col h-full"
                        >
                          <div className="flex items-center justify-between mb-8">
                            <div className="flex items-center gap-4">
                              <div
                                className={`w-12 h-12 rounded-2xl bg-gradient-to-br ${tierGradients[selectedMessage.tier]} flex items-center justify-center text-white shadow-xl`}
                              >
                                <MousePointer2 size={24} />
                              </div>
                              <div>
                                <h4 className="font-black text-sm uppercase tracking-widest">
                                  {selectedMessage.tier.replace("_", " ")}{" "}
                                  Deploy
                                </h4>
                                <p className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">
                                  {selectedMessage.channel} CHANNEL
                                </p>
                              </div>
                            </div>
                            <AppleButton
                              variant="secondary"
                              onClick={() =>
                                copyMessage(
                                  selectedMessage,
                                  strategy.messages.indexOf(selectedMessage),
                                )
                              }
                              size="sm"
                              className="font-bold border-border/40"
                            >
                              {copiedIndex ===
                              strategy.messages.indexOf(selectedMessage) ? (
                                <Check size={14} className="text-emerald-500" />
                              ) : (
                                <Copy size={14} />
                              )}
                              {copiedIndex ===
                              strategy.messages.indexOf(selectedMessage)
                                ? "Copied"
                                : "Copy Source"}
                            </AppleButton>
                          </div>

                          <div className="flex-1 mb-8 overflow-auto">
                            {selectedMessage.subject && (
                              <div className="mb-4 p-4 bg-secondary/30 rounded-2xl border border-border/20">
                                <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground block mb-1">
                                  Subject Protocol
                                </span>
                                <p className="font-bold text-sm">
                                  {selectedMessage.subject}
                                </p>
                              </div>
                            )}
                            <div className="p-6 bg-secondary/10 rounded-3xl text-sm font-medium leading-[1.8] min-h-[250px] whitespace-pre-wrap">
                              {selectedMessage.message}
                            </div>
                          </div>

                          <div className="pt-6 border-t border-border/20 space-y-4">
                            <h5 className="text-[10px] font-black uppercase tracking-widest text-emerald-600">
                              Engagement Tips
                            </h5>
                            <div className="grid gap-2">
                              {selectedMessage.tips.map((tip, i) => (
                                <div
                                  key={i}
                                  className="flex items-start gap-2 p-3 bg-emerald-500/5 rounded-xl border border-emerald-500/10 text-xs font-bold text-emerald-700 leading-relaxed"
                                >
                                  <Sparkles
                                    size={14}
                                    className="mt-0.5 shrink-0"
                                  />
                                  {tip}
                                </div>
                              ))}
                            </div>
                          </div>
                        </motion.div>
                      ) : (
                        <div className="flex-1 flex flex-col items-center justify-center text-center opacity-30">
                          <div className="w-20 h-20 rounded-[2rem] bg-secondary flex items-center justify-center mb-6">
                            <Rocket
                              size={40}
                              className="text-muted-foreground"
                            />
                          </div>
                          <h4 className="text-xl font-black mb-2 tracking-tight uppercase">
                            Awaiting Selection
                          </h4>
                          <p className="text-xs font-bold text-muted-foreground uppercase tracking-widest max-w-[200px]">
                            Select a tier from the escalation grid to view
                            tactial output
                          </p>
                        </div>
                      )}
                    </AppleCard>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="idle"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="h-full flex flex-col items-center justify-center py-40 opacity-20"
                >
                  <div className="w-32 h-32 rounded-[3rem] border-4 border-dashed border-border flex items-center justify-center mb-10">
                    <History size={64} />
                  </div>
                  <h3 className="text-3xl font-black tracking-tight uppercase mb-4">
                    Command Terminal Idle
                  </h3>
                  <p className="text-xs font-black uppercase tracking-[0.3em] max-w-sm text-center">
                    Protocol initiation required. Connect job objective and
                    profile data to begin synethsis of outreach matrix.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}
