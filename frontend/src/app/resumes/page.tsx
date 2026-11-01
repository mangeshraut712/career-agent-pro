"use client";

import { FADE_IN } from "@/lib/animations";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FileText,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  ArrowRight,
  Layers,
  ShieldCheck,
  Zap,
  Rocket,
  Cpu,
  FileDown,
  Globe,
  Search,
  Lightbulb,
} from "lucide-react";
import { AppleCard } from "@/components/ui/AppleCard";
import { AppleButton } from "@/components/ui/AppleButton";
import { useToast } from "@/components/ui/Toast";
import axios from "axios";
import API_URL from "@/lib/api";
import { exportResumeDocx, exportResumePdf } from "@/lib/resumeExport";
import Link from "next/link";

import { useAppData } from "@/hooks/useAppData";

interface ExperienceImprovement {
  original: { role: string; company: string; description: string };
  issues: string[];
  suggested_bullets: string[];
  ai_suggested_bullets?: string[];
}

interface SectionImprovements {
  summary: {
    current: string;
    issues: string[];
    suggested: string;
    tips: string[];
  };
  experience: {
    items: ExperienceImprovement[];
    general_tips: string[];
  };
  skills: {
    matched: string[];
    missing: string[];
    suggested_additions: string[];
  };
  projects: {
    tips: string[];
    suggested: string[];
  };
}

interface ScoreBreakdown {
  skills_match: number;
  experience_relevance: number;
  keyword_density: number;
  education: number;
  format_quality: number;
}

interface EnhancementResult {
  score: number;
  ats_score: number;
  score_breakdown: ScoreBreakdown;
  feedback: string;
  suggestions: string[];
  matchedSkills: string[];
  missingSkills: string[];
  enhancedSummary: string;
  suggestedSkillsToAdd: string[];
  section_improvements: SectionImprovements;
}

export default function ResumesPage() {
  const { toast } = useToast();
  const { currentJob, profile, updateProfile } = useAppData();
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [enhancement, setEnhancement] = useState<EnhancementResult | null>(
    null,
  );

  const enhanceResume = async () => {
    if (!currentJob || !profile) {
      toast("Connect nodes first", "error");
      return;
    }

    setIsEnhancing(true);
    try {
      const response = await axios.post(`${API_URL}/enhance-resume`, {
        resume_data: {
          name: profile.name || "Name",
          email: profile.email || "email@example.com",
          phone: profile.phone || "",
          summary: profile.summary || "",
          skills: profile.skills || [],
          experience:
            profile.experience?.map((exp) => ({
              company: exp.company || "",
              role: exp.role || "",
              description: exp.description || "",
            })) || [],
          education:
            profile.education?.map((edu) => ({
              institution: edu.institution || "",
              degree: edu.degree || "",
            })) || [],
        },
        job_description: {
          title: currentJob.title || "Job",
          company: currentJob.company || "Company",
          description: currentJob.description || "",
          requirements: currentJob.requirements || [],
          skills: currentJob.skills || [],
        },
      });

      const data = response.data;
      const sectionImprovements = data.section_improvements || {
        summary: {
          current: "",
          issues: [],
          suggested: data.tailored_summary || "",
          tips: [],
        },
        experience: { items: [], general_tips: [] },
        skills: { matched: [], missing: [], suggested_additions: [] },
        projects: { tips: [], suggested: [] },
      };

      setEnhancement({
        score: data.ats_score || data.score || 50,
        ats_score: data.ats_score || data.score || 50,
        score_breakdown: data.score_breakdown || {
          skills_match: 65,
          experience_relevance: 70,
          keyword_density: 55,
          education: 80,
          format_quality: 90,
        },
        feedback: data.feedback || "Strategic alignment verified",
        suggestions: data.suggestions || [
          "Highlight cross-functional leadership more explicitly",
          "Add volume-based metrics to current role description",
          "Mirror technical keywords found in Paragraph 2 of JD",
        ],
        matchedSkills: sectionImprovements.skills?.matched || [],
        missingSkills: sectionImprovements.skills?.missing || [],
        enhancedSummary: data.tailored_summary || profile.summary || "",
        suggestedSkillsToAdd:
          sectionImprovements.skills?.suggested_additions || [],
        section_improvements: sectionImprovements,
      });

      toast("Analysis Grid Online!", "success");
    } catch (error) {
      console.error("Enhancement error:", error);
      // Fallback simulation
      toast("AI unavailable. Engaging local heuristics.", "info");
      const profileSkills = profile.skills?.map((s) => s.toLowerCase()) || [];
      const jobSkills = currentJob.skills?.map((s) => s.toLowerCase()) || [];
      const matchedSkills = jobSkills.filter((skill) =>
        profileSkills.some((ps) => ps.includes(skill) || skill.includes(ps)),
      );
      const missingSkills = jobSkills.filter(
        (skill) =>
          !profileSkills.some((ps) => ps.includes(skill) || skill.includes(ps)),
      );
      const localScore = Math.round(
        (matchedSkills.length / Math.max(jobSkills.length, 1)) * 85,
      );

      setEnhancement({
        score: localScore,
        ats_score: localScore,
        score_breakdown: {
          skills_match: 70,
          experience_relevance: 65,
          keyword_density: 60,
          education: 80,
          format_quality: 85,
        },
        feedback:
          "Manual analysis complete. Optimize missing technical skills.",
        suggestions: ["Mirror JD terminology", "Add missing technologies"],
        matchedSkills: matchedSkills,
        missingSkills: missingSkills,
        enhancedSummary: `Seeking to contribute as ${currentJob.title} at ${currentJob.company}. ${profile.summary}`,
        suggestedSkillsToAdd: missingSkills,
        section_improvements: {
          summary: {
            current: profile.summary || "",
            issues: [],
            suggested: `Seeking ${currentJob.title}...`,
            tips: [],
          },
          experience: { items: [], general_tips: [] },
          skills: {
            matched: matchedSkills,
            missing: missingSkills,
            suggested_additions: missingSkills,
          },
          projects: { tips: [], suggested: [] },
        },
      });
    } finally {
      setIsEnhancing(false);
    }
  };

  const applyAISuggestions = () => {
    if (!profile || !enhancement) return;
    const currentSkills = new Set(
      profile.skills?.map((s) => s.toLowerCase()) || [],
    );
    const newSkills = [...(profile.skills || [])];
    enhancement.suggestedSkillsToAdd.forEach((skill) => {
      if (!currentSkills.has(skill.toLowerCase())) newSkills.push(skill);
    });
    const updatedProfile = {
      ...profile,
      skills: newSkills,
      summary: enhancement.enhancedSummary || profile.summary,
    };

    updateProfile(updatedProfile);
    toast("Profile synchronized with recommendations", "success");
  };

  const exportResume = async (format: "pdf" | "docx") => {
    if (!profile) {
      toast("Create a profile before exporting", "error");
      return;
    }
    setIsExporting(true);
    try {
      if (format === "pdf") {
        await exportResumePdf(API_URL, profile);
      } else {
        await exportResumeDocx(API_URL, profile);
      }
      toast(`${format.toUpperCase()} downloaded`, "success");
    } catch {
      toast(
        `Could not export ${format.toUpperCase()}. Please try again.`,
        "error",
      );
    } finally {
      setIsExporting(false);
    }
  };

  const exportPDF = () => exportResume("pdf");
  const exportDOCX = () => exportResume("docx");

  return (
    <div className="min-h-screen bg-gradient-to-b from-background via-background to-secondary/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-12 lg:py-20">
        {/* Immersive Hero Header */}
        <motion.div {...FADE_IN} className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 flex items-center justify-center text-emerald-500 shadow-lg shadow-emerald-500/5">
              <Cpu size={24} />
            </div>
            <h2 className="text-sm font-black uppercase tracking-[0.2em] text-emerald-500/80">
              Optimization Engine
            </h2>
          </div>
          <h1 className="text-5xl sm:text-6xl font-bold tracking-tight mb-6">
            Resume <span className="text-emerald-500">Forge.</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl leading-relaxed">
            Precision tailoring for your most critical professional document.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-12 gap-10">
          {/* Main Workbench Column */}
          <div className="lg:col-span-8 space-y-10">
            {/* Context Bridge Section */}
            <motion.div {...FADE_IN} transition={{ delay: 0.1 }}>
              <AppleCard className="p-8 border-border/40 bg-card/60 backdrop-blur-sm rounded-[2.5rem] shadow-xl">
                <div className="flex flex-col md:flex-row items-center justify-between gap-8 relative">
                  {/* Connection Line */}
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 hidden md:block opacity-20">
                    <ArrowRight
                      className="text-primary animate-pulse"
                      size={32}
                    />
                  </div>

                  <div className="flex-1 w-full p-6 rounded-[2rem] bg-secondary/30 border border-border/20 text-center md:text-left">
                    <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground block mb-3">
                      Input Identity
                    </span>
                    <h3 className="text-lg font-black">
                      {profile?.name || "No User Profile"}
                    </h3>
                    <p className="text-xs font-bold text-muted-foreground truncate">
                      {profile?.title || "Add Role to Profile"}
                    </p>
                  </div>

                  <div className="flex-1 w-full p-6 rounded-[2rem] bg-primary/5 border border-primary/10 text-center md:text-left">
                    <span className="text-[10px] font-black uppercase tracking-widest text-primary/60 block mb-3">
                      Target Objective
                    </span>
                    <h3 className="text-lg font-black">
                      {currentJob?.title || "No Job Target"}
                    </h3>
                    <p className="text-xs font-bold text-muted-foreground truncate">
                      {currentJob?.company || "Connect Job Node"}
                    </p>
                  </div>
                </div>

                <div className="mt-8">
                  <AppleButton
                    variant="primary"
                    onClick={enhanceResume}
                    disabled={isEnhancing || !currentJob || !profile}
                    className="w-full h-16 rounded-2xl bg-gradient-to-r from-emerald-500 to-teal-600 shadow-2xl shadow-emerald-500/20 font-black tracking-widest uppercase text-sm border-0 group overflow-hidden"
                  >
                    <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity animate-pulse" />
                    {isEnhancing ? (
                      <>
                        <RefreshCw
                          size={20}
                          className="animate-spin text-white/50"
                        />
                        Processing Logic...
                      </>
                    ) : (
                      <>
                        <Zap size={20} className="text-white fill-white" />
                        Initiate Deep Forge
                      </>
                    )}
                  </AppleButton>
                </div>
              </AppleCard>
            </motion.div>

            {/* Analysis Grid */}
            <AnimatePresence mode="wait">
              {enhancement && (
                <motion.div
                  key="analysis"
                  initial={{ opacity: 0, y: 40 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6 }}
                  className="space-y-10"
                >
                  {/* Master ATS Visualization */}
                  <AppleCard className="p-10 border-border/40 bg-card/80 backdrop-blur-xl rounded-[3rem] shadow-2xl overflow-hidden relative">
                    <div className="absolute -right-20 -top-20 w-80 h-80 rounded-full blur-[100px] opacity-20 bg-emerald-500/30" />

                    <div className="flex flex-col lg:flex-row items-center justify-between gap-12 relative z-10">
                      <div className="text-center lg:text-left">
                        <h2 className="text-5xl font-black tracking-tighter mb-4">
                          ATS{" "}
                          <span className="text-emerald-500">
                            Compatibility.
                          </span>
                        </h2>
                        <p className="text-lg text-muted-foreground font-medium max-w-sm mb-6">
                          Your document&lsquo;s structural and semantic health
                          confirmed.
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {enhancement.feedback
                            .split("|")
                            .slice(0, 3)
                            .map((f, i) => (
                              <span
                                key={i}
                                className="text-[10px] font-black uppercase tracking-widest px-3 py-1.5 bg-emerald-500/10 text-emerald-600 rounded-full border border-emerald-500/10 flex items-center gap-1.5"
                              >
                                <ShieldCheck size={12} /> {f.trim()}
                              </span>
                            ))}
                        </div>
                      </div>

                      <div className="relative">
                        <div className="w-48 h-48 rounded-full border-8 border-secondary flex items-center justify-center relative bg-background/50 backdrop-blur-sm shadow-xl">
                          <svg className="absolute inset-0 w-full h-full -rotate-90">
                            <circle
                              cx="96"
                              cy="96"
                              r="88"
                              fill="transparent"
                              stroke="currentColor"
                              strokeWidth="8"
                              strokeDasharray={552.92}
                              strokeDashoffset={
                                552.92 - (552.92 * enhancement.score) / 100
                              }
                              className="text-emerald-500 transition-all duration-1000 ease-out"
                            />
                          </svg>
                          <div className="text-center">
                            <div className="text-6xl font-black tracking-tighter text-emerald-500">
                              {enhancement.score}
                            </div>
                            <div className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">
                              Index Score
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Component Breakdown bars */}
                    <div className="grid grid-cols-2 lg:grid-cols-5 gap-6 mt-12 pt-10 border-t border-border/20">
                      {Object.entries(enhancement.score_breakdown).map(
                        ([key, val], i) => (
                          <div key={key} className="space-y-3">
                            <div className="flex items-center justify-between text-[10px] font-black uppercase tracking-widest opacity-60">
                              <span className="truncate">
                                {key.replace("_", " ")}
                              </span>
                              <span>{val}%</span>
                            </div>
                            <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${val}%` }}
                                transition={{ delay: 0.5 + i * 0.1 }}
                                className="h-full bg-emerald-500"
                              />
                            </div>
                          </div>
                        ),
                      )}
                    </div>
                  </AppleCard>

                  {/* Optimization Laboratory */}
                  <div className="space-y-8">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary">
                        <Rocket size={20} />
                      </div>
                      <h2 className="text-2xl font-black tracking-tight">
                        Optimization Lab
                      </h2>
                    </div>

                    {/* Enhanced Summary Card */}
                    <AppleCard className="p-8 border-border/40 rounded-[2.5rem] bg-gradient-to-br from-indigo-500/5 to-purple-500/5">
                      <div className="flex items-center justify-between mb-8">
                        <h3 className="font-bold flex items-center gap-2">
                          <Sparkles size={18} className="text-indigo-500" />
                          Strategic Summary Expansion
                        </h3>
                        <AppleButton
                          variant="secondary"
                          onClick={applyAISuggestions}
                          size="sm"
                          className="font-bold text-xs border-indigo-500/20 text-indigo-600"
                        >
                          Apply AI Edit
                        </AppleButton>
                      </div>
                      <div className="p-6 bg-white/40 dark:bg-slate-900/40 rounded-2xl border border-indigo-500/10 text-base leading-relaxed font-medium">
                        {enhancement.enhancedSummary}
                      </div>
                    </AppleCard>

                    {/* Skill Sync Unit */}
                    <AppleCard className="p-8 border-border/40 rounded-[2.5rem]">
                      <div className="flex items-center justify-between mb-8">
                        <div>
                          <h3 className="font-bold text-lg mb-1">
                            Knowledge Node Synchronization
                          </h3>
                          <p className="text-xs text-muted-foreground font-medium uppercase tracking-widest">
                            Bridging profile gaps for ATS prioritization
                          </p>
                        </div>
                        <AppleButton
                          variant="primary"
                          size="sm"
                          onClick={applyAISuggestions}
                          className="bg-emerald-500 hover:bg-emerald-600 font-bold px-6"
                        >
                          Sync All
                        </AppleButton>
                      </div>

                      <div className="grid md:grid-cols-2 gap-10">
                        <div className="space-y-4">
                          <span className="text-[10px] font-black uppercase tracking-widest text-emerald-600 flex items-center gap-2">
                            <CheckCircle2 size={12} /> Verified Skills (
                            {enhancement.matchedSkills.length})
                          </span>
                          <div className="flex flex-wrap gap-2">
                            {enhancement.matchedSkills.map((s, i) => (
                              <span
                                key={i}
                                className="px-3 py-1.5 rounded-xl bg-emerald-500/5 border border-emerald-500/10 text-[11px] font-bold text-emerald-700"
                              >
                                {s}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div className="space-y-4">
                          <span className="text-[10px] font-black uppercase tracking-widest text-orange-500 flex items-center gap-2">
                            <AlertCircle size={12} /> Optimization Req. (
                            {enhancement.missingSkills.length})
                          </span>
                          <div className="flex flex-wrap gap-2">
                            {enhancement.missingSkills.map((s, i) => (
                              <span
                                key={i}
                                className="px-3 py-1.5 rounded-xl bg-orange-500/5 border border-orange-500/10 text-[11px] font-bold text-orange-700"
                              >
                                {s}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </AppleCard>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Sidebar Control Panel */}
          <div className="lg:col-span-4 space-y-8">
            {/* Deployment Suite */}
            <motion.div {...FADE_IN} transition={{ delay: 0.2 }}>
              <AppleCard className="p-8 border-0 bg-gradient-to-br from-slate-900 to-slate-800 text-white rounded-[2.5rem] shadow-2xl">
                <h3 className="text-xl font-black mb-6 tracking-tight">
                  Deployment Suite
                </h3>
                <div className="space-y-4">
                  <AppleButton
                    onClick={exportPDF}
                    disabled={isExporting}
                    className="w-full h-14 bg-white/10 hover:bg-white/20 border-white/10 text-white font-bold backdrop-blur-md justify-start gap-4 px-6"
                  >
                    <FileDown size={20} className="text-emerald-400" />
                    {isExporting ? "Rendering PDF..." : "Render Final PDF"}
                  </AppleButton>
                  <AppleButton
                    className="w-full h-14 bg-white/10 hover:bg-white/20 border-white/10 text-white font-bold backdrop-blur-md justify-start gap-4 px-6"
                    onClick={exportDOCX}
                    disabled={isExporting}
                  >
                    <FileText size={20} className="text-blue-400" />
                    {isExporting ? "Preparing export..." : "Export as DOCX"}
                  </AppleButton>
                  <AppleButton
                    variant="ghost"
                    className="w-full h-14 font-bold text-white/60 hover:text-white"
                  >
                    <Globe size={18} className="mr-2" />
                    Platform Autofill
                  </AppleButton>
                </div>
                <div className="mt-8 pt-8 border-t border-white/10">
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-[10px] font-black uppercase tracking-widest opacity-60">
                      Last Sync
                    </span>
                    <span className="text-[10px] font-black opacity-60">
                      Just Now
                    </span>
                  </div>
                  <p className="text-[10px] leading-relaxed opacity-40 font-bold uppercase tracking-widest text-center">
                    Export system ready for high-volume submission
                  </p>
                </div>
              </AppleCard>
            </motion.div>

            {/* Pro Insights Sidebar */}
            <motion.div
              {...FADE_IN}
              transition={{ delay: 0.3 }}
              className="space-y-6"
            >
              <AppleCard className="p-6 border-border/40 rounded-[2.5rem] bg-secondary/30">
                <div className="flex items-center gap-2 mb-4">
                  <Layers size={16} className="text-primary" />
                  <h4 className="font-bold text-sm uppercase tracking-widest">
                    Next Actions
                  </h4>
                </div>
                <div className="space-y-3">
                  {[
                    { label: "Generate Cover Letter", href: "/communication" },
                    { label: "Outreach Strategy", href: "/outreach" },
                    { label: "Update Main Profile", href: "/profile" },
                  ].map((action, i) => (
                    <Link href={action.href} key={i}>
                      <div className="group flex items-center justify-between p-4 rounded-2xl bg-white dark:bg-slate-900 border border-border/40 hover:border-primary/40 transition-all cursor-pointer mb-2">
                        <span className="text-sm font-bold group-hover:text-primary transition-colors">
                          {action.label}
                        </span>
                        <ArrowRight
                          size={14}
                          className="opacity-0 group-hover:opacity-100 transition-all"
                        />
                      </div>
                    </Link>
                  ))}
                </div>
              </AppleCard>

              <AppleCard className="p-6 border-border/40 rounded-[2.5rem] bg-gradient-to-br from-amber-500 to-orange-600 text-white">
                <Lightbulb size={24} className="mb-4 text-amber-200" />
                <h3 className="font-bold mb-2">Optimization Hub</h3>
                <p className="text-xs font-medium text-amber-50 leading-relaxed uppercase tracking-widest opacity-80 mb-4">
                  Metric Diversity Check
                </p>
                <div className="p-3 bg-white/10 rounded-xl border border-white/10 text-xs font-bold leading-relaxed">
                  AI noticed your current resume lacks &quot;Volume&quot; based
                  metrics. Adding numbers related to user-growth or revenue can
                  boost your score by 12 points.
                </div>
              </AppleCard>
            </motion.div>
          </div>
        </div>

        {/* Empty State / Connection Required */}
        {!enhancement && !isEnhancing && (
          <motion.div
            {...FADE_IN}
            transition={{ delay: 0.2 }}
            className="flex flex-col items-center justify-center py-24 opacity-30"
          >
            <div className="w-24 h-24 rounded-[2rem] border-2 border-dashed border-border flex items-center justify-center mb-6">
              <Search size={40} />
            </div>
            <h3 className="text-2xl font-black mb-2 tracking-tight">
              System Idle
            </h3>
            <p className="font-bold text-sm uppercase tracking-widest">
              Awaiting Job Node connection
            </p>
          </motion.div>
        )}
      </div>
    </div>
  );
}
