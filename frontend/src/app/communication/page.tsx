"use client";

import { FADE_IN } from "@/lib/animations";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mail,
  Linkedin,
  MessageSquare,
  Copy,
  RefreshCw,
  FileText,
  Layout,
  PenTool,
  Zap,
  ShieldCheck,
  History,
  Wand2,
  Check,
  Share2,
  Target,
} from "lucide-react";
import { AppleCard } from "@/components/ui/AppleCard";
import { AppleButton } from "@/components/ui/AppleButton";
import { useToast } from "@/components/ui/Toast";
import axios from "axios";
import API_URL from "@/lib/api";
import Link from "next/link";

import { useAppData } from "@/hooks/useAppData";

type MessageType = "cover_letter" | "email" | "linkedin" | "follow_up";

import type { LucideIcon } from "lucide-react";

const templates: Record<
  MessageType,
  {
    icon: LucideIcon;
    label: string;
    color: string;
    description: string;
    gradient: string;
  }
> = {
  cover_letter: {
    icon: FileText,
    label: "Cover Narrative",
    color: "bg-purple-500",
    gradient: "from-purple-500 to-indigo-600",
    description: "Strategic professional introduction",
  },
  email: {
    icon: Mail,
    label: "Direct Outreach",
    color: "bg-blue-500",
    gradient: "from-blue-500 to-cyan-500",
    description: "Recruiter-first email briefing",
  },
  linkedin: {
    icon: Linkedin,
    label: "Network Pulse",
    color: "bg-[#0a66c2]",
    gradient: "from-[#0a66c2] to-[#0077b5]",
    description: "Connection request protocol",
  },
  follow_up: {
    icon: MessageSquare,
    label: "Momentum Lock",
    color: "bg-emerald-500",
    gradient: "from-emerald-500 to-teal-600",
    description: "Post-engagement follow-up",
  },
};

export default function CommunicationPage() {
  const { toast } = useToast();
  const [messageType, setMessageType] = useState<MessageType>("cover_letter");
  const [isGenerating, setIsGenerating] = useState(false);
  const [copied, setCopied] = useState(false);
  const [generatedContent, setGeneratedContent] = useState("");
  const { currentJob, profile } = useAppData();

  const generateMessage = async () => {
    if (!currentJob) {
      toast("Select a target job first", "error");
      return;
    }

    setIsGenerating(true);
    try {
      const resumePayload = {
        name: profile?.name || "",
        email: profile?.email || "",
        summary: profile?.summary || "",
        skills: profile?.skills || [],
        experience: profile?.experience || [],
      };

      const jobPayload = {
        title: currentJob.title,
        company: currentJob.company,
        description: currentJob.description || "",
        requirements: currentJob.requirements || [],
        skills: currentJob.skills || [],
      };

      if (messageType === "cover_letter") {
        const response = await axios.post(`${API_URL}/generate-cover-letter`, {
          resume_data: resumePayload,
          job_description: jobPayload,
          template_type: "apply_pilot",
        });
        setGeneratedContent(
          response.data.content || "Generating localized narrative...",
        );
      } else {
        const typeMap: Record<MessageType, string> = {
          cover_letter: "email",
          email: "email",
          linkedin: "linkedin_message",
          follow_up: "follow_up",
        };
        const response = await axios.post(`${API_URL}/generate-communication`, {
          resume_data: resumePayload,
          job_description: jobPayload,
          type: typeMap[messageType],
        });
        setGeneratedContent(response.data.content);
      }
      toast("Protocol Generated.", "success");
    } catch (error: unknown) {
      console.error("Generation error:", error);
      // Use fallback templates when AI fails
      const name = profile?.name || "Your Name";
      const title = currentJob.title || "the position";
      const company = currentJob.company || "your company";
      const skills =
        profile?.skills?.slice(0, 3).join(", ") || "relevant skills";

      const fallbackTemplates: Record<MessageType, string> = {
        cover_letter: `Dear Hiring Manager,

I am writing to express my strong interest in the ${title} position at ${company}.

With my background in ${skills}, I am confident I can contribute meaningfully to your team. My experience has prepared me well for this role, and I am excited about the opportunity to bring my skills to ${company}.

I would welcome the opportunity to discuss how my qualifications align with your needs.

Best regards,
${name}`,
        email: `Subject: Application for ${title}

Dear Hiring Manager,

I am interested in the ${title} position at ${company}. My expertise in ${skills} makes me a strong candidate for this role.

I would appreciate the opportunity to discuss this position further.

Best regards,
${name}`,
        linkedin: `Hi! I noticed the ${title} opening at ${company} and I'm very interested. My background in ${skills} aligns well with what you're looking for. Would love to connect and learn more about the team!`,
        follow_up: `Subject: Following Up - ${title} Application

Dear Hiring Manager,

I wanted to follow up on my application for the ${title} position at ${company}. I remain very interested in this opportunity and am happy to provide any additional information.

Looking forward to hearing from you.

Best regards,
${name}`,
      };

      setGeneratedContent(fallbackTemplates[messageType]);
      toast(
        "Using template mode. AI enhancement available when connected.",
        "info",
      );
    } finally {
      setIsGenerating(false);
    }
  };

  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(generatedContent);
    setCopied(true);
    toast("Copied to Clipboard!", "success");
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background via-background to-secondary/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-12 lg:py-20">
        {/* Immersive Header */}
        <motion.div {...FADE_IN} className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-2xl bg-purple-500/10 flex items-center justify-center text-purple-500 shadow-lg shadow-purple-500/5">
              <PenTool size={24} />
            </div>
            <h2 className="text-sm font-black uppercase tracking-[0.2em] text-purple-500/80">
              Voice Studio
            </h2>
          </div>
          <h1 className="text-5xl sm:text-6xl font-bold tracking-tight mb-6">
            Message <span className="text-purple-500">Forge.</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl leading-relaxed">
            Precision-crafted narratives designed to bridge the gap between your
            identity and their requirements.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-12 gap-10">
          {/* Control Sidebar */}
          <div className="lg:col-span-4 space-y-8">
            <motion.div {...FADE_IN} transition={{ delay: 0.1 }}>
              <AppleCard className="p-8 border-border/40 bg-card/60 backdrop-blur-sm rounded-[2.5rem] shadow-xl">
                <div className="flex items-center gap-2 mb-6">
                  <Layout size={16} className="text-primary" />
                  <h3 className="text-xs font-black uppercase tracking-widest text-muted-foreground">
                    Select Protocol
                  </h3>
                </div>
                <div className="space-y-3">
                  {(Object.keys(templates) as MessageType[]).map((type) => {
                    const template = templates[type];
                    const isActive = messageType === type;
                    return (
                      <motion.button
                        key={type}
                        whileHover={{ x: 6, scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => {
                          setMessageType(type);
                          setGeneratedContent("");
                        }}
                        className={`w-full flex items-center gap-4 p-5 rounded-2xl transition-all ${
                          isActive
                            ? `bg-gradient-to-r ${template.gradient} text-white shadow-lg shadow-primary/10`
                            : "hover:bg-secondary/80 text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        <div
                          className={`w-8 h-8 rounded-xl ${isActive ? "bg-white/20" : template.color} flex items-center justify-center`}
                        >
                          <template.icon
                            size={16}
                            className={isActive ? "text-white" : "text-white"}
                          />
                        </div>
                        <div className="text-left">
                          <p className="text-sm font-black tracking-tight">
                            {template.label}
                          </p>
                          <p
                            className={`text-[10px] font-bold uppercase tracking-widest opacity-60 ${isActive ? "text-white" : ""}`}
                          >
                            {template.description}
                          </p>
                        </div>
                      </motion.button>
                    );
                  })}
                </div>
              </AppleCard>
            </motion.div>

            <motion.div {...FADE_IN} transition={{ delay: 0.2 }}>
              <AppleCard className="p-8 border-border/40 rounded-[2.5rem]">
                <div className="flex items-center gap-2 mb-6">
                  <Target size={16} className="text-primary" />
                  <h3 className="text-xs font-black uppercase tracking-widest text-muted-foreground">
                    Active Connection
                  </h3>
                </div>
                {currentJob ? (
                  <div className="p-6 bg-secondary/30 rounded-3xl border border-border/20">
                    <h4 className="font-black text-lg mb-1">
                      {currentJob.title}
                    </h4>
                    <p className="text-sm font-bold text-muted-foreground mb-4">
                      {currentJob.company}
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {currentJob.skills?.slice(0, 3).map((s, i) => (
                        <span
                          key={i}
                          className="px-2.5 py-1 bg-white dark:bg-slate-800 rounded-lg text-[10px] font-black uppercase tracking-widest border border-border/40"
                        >
                          {s}
                        </span>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-6">
                    <p className="text-xs font-bold text-muted-foreground uppercase tracking-widest leading-relaxed">
                      System requires Job Node connection
                    </p>
                    <Link
                      href="/jobs"
                      className="mt-4 block text-xs font-black text-primary hover:underline"
                    >
                      BROWSE JOBS
                    </Link>
                  </div>
                )}
              </AppleCard>
            </motion.div>
          </div>

          {/* Creative Canvas */}
          <div className="lg:col-span-8 flex flex-col">
            <motion.div
              {...FADE_IN}
              transition={{ delay: 0.3 }}
              className="h-full flex flex-col"
            >
              <AppleCard className="flex-1 p-0 border-border/40 bg-card/80 backdrop-blur-xl rounded-[3rem] shadow-2xl overflow-hidden flex flex-col min-h-[600px]">
                {/* Canvas Toolbar */}
                <div className="p-6 border-b border-border/20 flex items-center justify-between bg-secondary/10">
                  <div className="flex items-center gap-4">
                    <div
                      className={`w-10 h-10 rounded-xl bg-gradient-to-r ${templates[messageType].gradient} flex items-center justify-center text-white shadow-lg shadow-primary/10`}
                    >
                      {(() => {
                        const Icon = templates[messageType].icon;
                        return <Icon size={20} />;
                      })()}
                    </div>
                    <div>
                      <h3 className="font-black tracking-tight">
                        {templates[messageType].label}
                      </h3>
                      <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground/60">
                        Creative Canvas
                      </span>
                    </div>
                  </div>

                  {generatedContent && (
                    <div className="flex gap-3">
                      <AppleButton
                        variant="secondary"
                        onClick={copyToClipboard}
                        size="sm"
                        className="font-bold border-border/40 backdrop-blur-sm"
                      >
                        {copied ? (
                          <Check size={16} className="text-emerald-500" />
                        ) : (
                          <Copy size={16} />
                        )}
                        {copied ? "Synchronized" : "Copy Source"}
                      </AppleButton>
                      <AppleButton
                        variant="secondary"
                        size="sm"
                        className="font-bold border-border/40"
                      >
                        <Share2 size={16} />
                      </AppleButton>
                    </div>
                  )}
                </div>

                {/* Main Editor Area */}
                <div className="flex-1 relative p-10 overflow-auto">
                  <AnimatePresence mode="wait">
                    {isGenerating ? (
                      <motion.div
                        key="thinking"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute inset-0 flex flex-col items-center justify-center bg-card/10 backdrop-blur-sm z-10"
                      >
                        <div className="text-center space-y-6">
                          <div className="relative">
                            <div className="w-20 h-20 rounded-[2rem] bg-purple-500/10 flex items-center justify-center text-purple-500 mx-auto">
                              <Wand2 size={40} className="animate-pulse" />
                            </div>
                            <div className="absolute top-0 right-0">
                              <RefreshCw
                                size={24}
                                className="text-purple-400 animate-spin"
                              />
                            </div>
                          </div>
                          <div className="space-y-2">
                            <h3 className="text-xl font-black tracking-tight">
                              Synthesizing Narrative...
                            </h3>
                            <p className="text-[10px] font-black uppercase tracking-[0.2em] text-muted-foreground animate-pulse">
                              Analyzing Job Context • Aligning Profile Gaps
                            </p>
                          </div>
                        </div>
                      </motion.div>
                    ) : generatedContent ? (
                      <motion.div
                        key="content"
                        initial={{ opacity: 0, scale: 0.98 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="h-full"
                      >
                        <textarea
                          value={generatedContent}
                          onChange={(e) => setGeneratedContent(e.target.value)}
                          className="w-full h-full min-h-[500px] bg-transparent resize-none text-lg font-medium leading-[1.8] focus:outline-none placeholder:text-muted-foreground/20 selection:bg-purple-500/10"
                          placeholder="The AI narrative will appear here..."
                        />
                      </motion.div>
                    ) : (
                      <motion.div
                        key="empty"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="h-full flex flex-col items-center justify-center text-center space-y-8 py-20"
                      >
                        <div className="w-24 h-24 rounded-[2.5rem] bg-secondary/50 flex items-center justify-center">
                          <MessageSquare
                            size={48}
                            className="text-muted-foreground opacity-20"
                          />
                        </div>
                        <div className="space-y-4 max-w-sm">
                          <h3 className="text-2xl font-black tracking-tight">
                            Studio Awaiting Input
                          </h3>
                          <p className="text-sm font-bold text-muted-foreground leading-relaxed">
                            Select your communication protocol and initiate the
                            forge to begin crafting your narrative.
                          </p>
                        </div>
                        <AppleButton
                          onClick={generateMessage}
                          disabled={!currentJob}
                          className="h-14 px-10 bg-gradient-to-r from-purple-500 to-indigo-600 font-black tracking-widest uppercase text-xs shadow-xl shadow-purple-500/20"
                        >
                          <Zap size={18} className="mr-2 fill-white" />
                          Initialize Studio
                        </AppleButton>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* Status Bar */}
                {generatedContent && (
                  <div className="px-8 py-5 border-t border-border/20 bg-secondary/5 flex items-center justify-between">
                    <div className="flex items-center gap-6">
                      <div className="flex items-center gap-2">
                        <ShieldCheck size={14} className="text-emerald-500" />
                        <span className="text-[10px] font-black uppercase tracking-widest opacity-60">
                          Strategic Alignment Verified
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <History size={14} className="text-purple-500" />
                        <span className="text-[10px] font-black uppercase tracking-widest opacity-60">
                          {generatedContent.split(/\s+/).length} Words Synced
                        </span>
                      </div>
                    </div>
                    <button
                      onClick={generateMessage}
                      className="text-[10px] font-black uppercase tracking-widest text-primary hover:text-purple-600 transition-colors flex items-center gap-1.5"
                    >
                      <RefreshCw size={12} /> Re-Synthesize
                    </button>
                  </div>
                )}
              </AppleCard>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
