"use client";

import { FADE_IN } from "@/lib/animations";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    Target, Sparkles, CheckCircle2, AlertTriangle, XCircle,
    TrendingUp, TrendingDown, ArrowRight, RefreshCw,
    Briefcase, Award, BarChart3, Zap, ShieldCheck,
    PieChart, Layers, Info, CheckCircle
} from "lucide-react";
import { AppleCard } from "@/components/ui/AppleCard";
import { AppleButton } from "@/components/ui/AppleButton";
import { useToast } from "@/components/ui/Toast";
import axios from "axios";
import API_URL from "@/lib/api";
import { useAppData } from "@/hooks/useAppData";
import Link from "next/link";

interface CompetencyMatch {
    area: string;
    score: number;
    weight: number;
    matched: string[];
    missing: string[];
}

interface AssessmentResult {
    fit_score: number;
    fit_level: string;
    strengths: string[];
    gaps: string[];
    spinning_recommendation: string;
    action_items: string[];
    resume_distribution: Record<string, number>;
    competency_breakdown: CompetencyMatch[];
}



type FitLevelName = "excellent" | "strong" | "moderate" | "weak";

const FALLBACK_FIT_THEME = {
    color: "text-muted-foreground",
    bg: "bg-secondary",
    border: "border-border",
    gradient: "from-secondary to-muted",
};

function isFitLevel(level: string): level is FitLevelName {
    return level === "excellent" || level === "strong" || level === "moderate" || level === "weak";
}

const getFitTheme = (level: string) => {
    const normalized = level.toLowerCase();
    if (!isFitLevel(normalized)) {
        return FALLBACK_FIT_THEME;
    }
    switch (normalized) {
        case "excellent": return { color: "text-emerald-500", bg: "bg-emerald-500/10", border: "border-emerald-500/20", gradient: "from-emerald-500 to-teal-400" };
        case "strong": return { color: "text-blue-500", bg: "bg-blue-500/10", border: "border-blue-500/20", gradient: "from-blue-500 to-indigo-400" };
        case "moderate": return { color: "text-amber-500", bg: "bg-amber-500/10", border: "border-amber-500/20", gradient: "from-amber-500 to-orange-400" };
        case "weak": return { color: "text-rose-500", bg: "bg-rose-500/10", border: "border-rose-500/20", gradient: "from-rose-500 to-pink-400" };
        default: {
            const unhandled: never = normalized;
            return unhandled;
        }
    }
};

export default function FitAnalysisPage() {
    const { toast } = useToast();
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const { currentJob, profile } = useAppData();
    const [assessment, setAssessment] = useState<AssessmentResult | null>(null);
    const [analysisError, setAnalysisError] = useState<string | null>(null);

    const analyzeJobFit = async () => {
        if (!currentJob) {
            toast("Please analyze a job first", "error");
            return;
        }

        setIsAnalyzing(true);
        setAssessment(null);
        setAnalysisError(null);

        try {
            const response = await axios.post(`${API_URL}/assess-job-fit`, {
                job_description: `${currentJob.title} at ${currentJob.company}\n\n${currentJob.description}\n\nRequirements: ${currentJob.requirements?.join(", ")}\n\nSkills: ${currentJob.skills?.join(", ")}`,
                resume_data: {
                    name: profile?.name || "",
                    email: profile?.email || "",
                    summary: profile?.summary || "",
                    skills: profile?.skills || [],
                    experience: profile?.experience || [],
                }
            });

            setAssessment(response.data);
            toast("Analysis complete", "success");
        } catch (error) {
            console.error("Analysis error:", error);
            setAnalysisError("Fit analysis could not be completed. Check that the API is running, then try again. No score was invented.");
            toast("Fit analysis failed. No score was generated.", "error");
        } finally {
            setIsAnalyzing(false);
        }
    };

    const theme = assessment ? getFitTheme(assessment.fit_level) : getFitTheme("");

    return (
        <div className="min-h-screen bg-gradient-to-b from-background via-background to-secondary/10">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 py-12 lg:py-20">

                {/* Header branding */}
                <motion.div {...FADE_IN} className="mb-16">
                    <div className="flex items-center gap-3 mb-6">
                        <div className="w-12 h-12 rounded-2xl bg-orange-500/10 flex items-center justify-center text-orange-500 shadow-lg shadow-orange-500/5">
                            <Target size={24} />
                        </div>
                        <h2 className="text-sm font-black uppercase tracking-[0.2em] text-orange-500/80">Fit Intelligence</h2>
                    </div>
                    <h1 className="text-5xl sm:text-6xl font-bold tracking-tight mb-6">
                        Strategic <span className="text-orange-500">Alignment.</span>
                    </h1>
                    <p className="text-xl text-muted-foreground max-w-2xl leading-relaxed">
                        Precision-engineered analysis of your experience against market demands.
                    </p>
                </motion.div>

                {/* Pre-Analysis Status Grid */}
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
                    <motion.div {...FADE_IN} transition={{ delay: 0.1 }} className="lg:col-span-1">
                        <AppleCard className="h-full p-8 border-border/40 bg-card/50 backdrop-blur-sm shadow-xl shadow-primary/5 rounded-[2rem]">
                            <div className="flex items-center justify-between mb-8">
                                <h3 className="font-bold flex items-center gap-2">
                                    <Briefcase className="text-primary" size={18} />
                                    Target JD
                                </h3>
                                <Link href="/jobs" className="text-[10px] font-black uppercase tracking-widest text-primary/60 hover:text-primary transition-colors">Change</Link>
                            </div>
                            {currentJob ? (
                                <div className="space-y-4">
                                    <div className="w-14 h-14 rounded-2xl bg-secondary flex items-center justify-center font-black text-xl text-primary border border-border/20">
                                        {currentJob.company?.charAt(0)}
                                    </div>
                                    <div>
                                        <p className="font-black text-lg leading-tight mb-1">{currentJob.title}</p>
                                        <p className="text-sm font-medium text-muted-foreground">{currentJob.company}</p>
                                    </div>
                                </div>
                            ) : (
                                <div className="py-6 text-center border-2 border-dashed border-border/40 rounded-3xl">
                                    <p className="text-xs font-bold text-muted-foreground/60 mb-4">No Data Input</p>
                                    <Link href="/jobs">
                                        <AppleButton size="sm" variant="outline" className="font-bold">Select Job</AppleButton>
                                    </Link>
                                </div>
                            )}
                        </AppleCard>
                    </motion.div>

                    <motion.div {...FADE_IN} transition={{ delay: 0.2 }} className="lg:col-span-1">
                        <AppleCard className="h-full p-8 border-border/40 bg-card/50 backdrop-blur-sm shadow-xl shadow-primary/5 rounded-[2rem]">
                            <div className="flex items-center justify-between mb-8">
                                <h3 className="font-bold flex items-center gap-2">
                                    <Award className="text-purple-500" size={18} />
                                    Candidate ID
                                </h3>
                                <Link href="/profile" className="text-[10px] font-black uppercase tracking-widest text-purple-500/60 hover:text-purple-500 transition-colors">Update</Link>
                            </div>
                            {profile ? (
                                <div className="space-y-4">
                                    <div className="w-14 h-14 rounded-2xl bg-purple-500/10 flex items-center justify-center font-black text-xl text-purple-500 border border-purple-500/20">
                                        {profile.name?.charAt(0) || "?"}
                                    </div>
                                    <div>
                                        <p className="font-black text-lg leading-tight mb-1">{profile.name || "Unnamed profile"}</p>
                                        <p className="text-sm font-medium text-muted-foreground truncate">{profile.skills?.length || 0} Loaded Competencies</p>
                                    </div>
                                </div>
                            ) : (
                                <div className="py-6 text-center border-2 border-dashed border-border/40 rounded-3xl">
                                    <p className="text-xs font-bold text-muted-foreground/60 mb-4">Profile Empty</p>
                                    <Link href="/profile">
                                        <AppleButton size="sm" variant="outline" className="font-bold text-purple-500 border-purple-500/20">Load Profile</AppleButton>
                                    </Link>
                                </div>
                            )}
                        </AppleCard>
                    </motion.div>

                    <motion.div {...FADE_IN} transition={{ delay: 0.3 }} className="lg:col-span-1">
                        <AppleButton
                            variant="primary"
                            onClick={analyzeJobFit}
                            disabled={isAnalyzing || !currentJob || !profile}
                            className="w-full h-full rounded-[2rem] p-8 bg-gradient-to-br from-orange-500 to-red-600 shadow-2xl shadow-orange-500/20 group relative overflow-hidden flex flex-col items-center justify-center gap-4 border-0"
                        >
                            {/* Animated background pulse */}
                            <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity animate-pulse" />

                            {isAnalyzing ? (
                                <>
                                    <RefreshCw size={48} className="animate-spin text-white/50" />
                                    <span className="text-lg font-black tracking-widest uppercase">Analyzing...</span>
                                </>
                            ) : (
                                <>
                                    <Zap size={48} className="text-white group-hover:scale-110 transition-transform" />
                                    <span className="text-lg font-black tracking-widest uppercase">Start Engine</span>
                                </>
                            )}
                        </AppleButton>
                    </motion.div>
                </div>

                {/* Analysis Results Display */}
                <AnimatePresence mode="wait">
                    {assessment && (
                        <motion.div
                            key="results"
                            initial={{ opacity: 0, y: 40 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            transition={{ duration: 0.6 }}
                            className="space-y-10"
                        >
                            {/* Master Score Display */}
                            <AppleCard className="p-10 border-border/40 bg-card/80 backdrop-blur-xl rounded-[3rem] shadow-2xl shadow-primary/5 relative overflow-hidden">
                                {/* Gradient mesh background */}
                                <div className={`absolute -right-20 -top-20 w-80 h-80 rounded-full blur-[100px] opacity-20 bg-gradient-to-br ${theme.gradient}`} />

                                <div className="flex flex-col lg:flex-row items-center justify-between gap-12 relative z-10">
                                    <div className="text-center lg:text-left">
                                        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-secondary mb-6">
                                            <ShieldCheck size={16} className={theme.color} />
                                            <span className="text-xs font-black uppercase tracking-widest">Analysis Confirmed</span>
                                        </div>
                                        <h2 className="text-5xl font-black tracking-tight mb-4">Overall <span className={theme.color}>Fit Score.</span></h2>
                                        <p className="text-xl text-muted-foreground font-medium max-w-sm">
                                            Your profile suggests a <span className={`font-black ${theme.color}`}>{assessment.fit_level}</span> alignment with this role.
                                        </p>
                                    </div>

                                    <div className="relative">
                                        {/* Radial Progress Visualization */}
                                        <div className="w-56 h-56 rounded-full border-8 border-secondary flex items-center justify-center shadow-lg relative bg-background/50 backdrop-blur-sm">
                                            <svg className="absolute inset-0 w-full h-full -rotate-90">
                                                <circle
                                                    cx="112"
                                                    cy="112"
                                                    r="104"
                                                    fill="transparent"
                                                    stroke="currentColor"
                                                    strokeWidth="8"
                                                    strokeDasharray={653.45}
                                                    strokeDashoffset={653.45 - (653.45 * assessment.fit_score) / 100}
                                                    className={`${theme.color} transition-all duration-1000 ease-out`}
                                                />
                                            </svg>
                                            <div className="text-center">
                                                <div className={`text-7xl font-black tracking-tighter ${theme.color}`}>{assessment.fit_score}</div>
                                                <div className="text-[10px] font-black uppercase tracking-[0.2em] text-muted-foreground">Percent Match</div>
                                            </div>
                                        </div>
                                        {/* Decorative elements around score */}
                                        <div className="absolute -top-4 -right-4 w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center animate-bounce">
                                            <Sparkles className="text-primary" size={20} />
                                        </div>
                                    </div>
                                </div>
                            </AppleCard>

                            <div className="grid lg:grid-cols-12 gap-10">
                                {/* Competency Breakdown Main Panel */}
                                <div className="lg:col-span-8 space-y-10">
                                    <AppleCard className="p-10 border-border/40 rounded-[2.5rem] bg-card/60">
                                        <div className="flex items-center gap-3 mb-10">
                                            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary">
                                                <PieChart size={20} />
                                            </div>
                                            <h3 className="text-xl font-bold tracking-tight">Competency Breakdown</h3>
                                        </div>

                                        <div className="space-y-8">
                                            {assessment.competency_breakdown.map((comp, i) => (
                                                <motion.div
                                                    key={comp.area}
                                                    initial={{ opacity: 0, x: -20 }}
                                                    animate={{ opacity: 1, x: 0 }}
                                                    transition={{ delay: i * 0.1 }}
                                                    className="group"
                                                >
                                                    <div className="flex items-center justify-between mb-3 font-bold">
                                                        <div className="flex items-center gap-3">
                                                            <div className={`w-2 h-2 rounded-full ${comp.score >= 75 ? "bg-emerald-500" : comp.score >= 50 ? "bg-amber-500" : "bg-rose-500"}`} />
                                                            <span className="text-lg">{comp.area}</span>
                                                            <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground bg-secondary/80 px-2 py-0.5 rounded-md">Weight: {comp.weight}%</span>
                                                        </div>
                                                        <span className="text-xl font-black">{comp.score}%</span>
                                                    </div>
                                                    <div className="w-full h-3 bg-secondary rounded-full overflow-hidden border border-border/10">
                                                        <motion.div
                                                            initial={{ width: 0 }}
                                                            animate={{ width: `${comp.score}%` }}
                                                            transition={{ duration: 1, delay: i * 0.15, ease: "easeOut" }}
                                                            className={`h-full rounded-full bg-gradient-to-r ${comp.score >= 75 ? "from-emerald-500 to-teal-400" :
                                                                comp.score >= 50 ? "from-amber-500 to-orange-400" :
                                                                    "from-rose-500 to-pink-400"
                                                                }`}
                                                        />
                                                    </div>
                                                    <div className="flex flex-wrap gap-2 mt-4">
                                                        {comp.matched.map((m, j) => (
                                                            <span key={j} className="text-[10px] font-bold px-3 py-1 bg-emerald-500/5 text-emerald-600 border border-emerald-500/10 rounded-full flex items-center gap-1">
                                                                <CheckCircle size={10} /> {m}
                                                            </span>
                                                        ))}
                                                        {comp.missing.map((m, j) => (
                                                            <span key={j} className="text-[10px] font-bold px-3 py-1 bg-rose-500/5 text-rose-500 border border-rose-500/10 rounded-full flex items-center gap-1 opacity-70">
                                                                <XCircle size={10} /> {m}
                                                            </span>
                                                        ))}
                                                    </div>
                                                </motion.div>
                                            ))}
                                        </div>
                                    </AppleCard>

                                    {/* Action items grid */}
                                    <div className="grid md:grid-cols-2 gap-8">
                                        <AppleCard className="p-8 border-border/40 rounded-[2.5rem] bg-emerald-500/[0.02]">
                                            <h3 className="font-bold flex items-center gap-2 text-emerald-600 mb-6">
                                                <TrendingUp size={20} /> Competitive Edge
                                            </h3>
                                            <div className="space-y-4">
                                                {assessment.strengths.map((str, i) => (
                                                    <div key={i} className="flex gap-3 text-sm leading-snug font-medium p-3 bg-card text-foreground border border-emerald-500/10 rounded-2xl">
                                                        <CheckCircle2 size={16} className="text-emerald-500 shrink-0" />
                                                        {str}
                                                    </div>
                                                ))}
                                            </div>
                                        </AppleCard>

                                        <AppleCard className="p-8 border-border/40 rounded-[2.5rem] bg-rose-500/[0.02]">
                                            <h3 className="font-bold flex items-center gap-2 text-rose-500 mb-6">
                                                <TrendingDown size={20} /> Strategic Gaps
                                            </h3>
                                            <div className="space-y-4">
                                                {assessment.gaps.map((gap, i) => (
                                                    <div key={i} className="flex gap-3 text-sm leading-snug font-medium p-3 bg-card text-foreground border border-rose-500/10 rounded-2xl">
                                                        <AlertTriangle size={16} className="text-rose-500 shrink-0" />
                                                        {gap}
                                                    </div>
                                                ))}
                                            </div>
                                        </AppleCard>
                                    </div>
                                </div>

                                {/* AI Strategy Sidebar */}
                                <div className="lg:col-span-4 space-y-10">
                                    <AppleCard className="p-8 border-0 bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-600 text-white rounded-[2.5rem] shadow-2xl shadow-purple-500/20">
                                        <Zap size={32} className="mb-6 text-yellow-300" />
                                        <h3 className="text-2xl font-black mb-4">Spin Strategy</h3>
                                        <p className="text-indigo-50/90 leading-relaxed font-medium mb-8">
                                            {assessment.spinning_recommendation}
                                        </p>
                                        <div className="p-5 bg-white/10 backdrop-blur-md rounded-2xl border border-white/20">
                                            <div className="flex items-center gap-2 text-xs font-black uppercase tracking-widest mb-3 opacity-80">
                                                <Info size={12} /> Pro Deployment
                                            </div>
                                            <p className="text-xs leading-relaxed font-bold">
                                                Use these keywords in paragraphs 2 and 3 of your profile summary to trigger ATS recognition.
                                            </p>
                                        </div>
                                    </AppleCard>

                                    <AppleCard className="p-8 border-border/40 rounded-[2.5rem] shadow-xl">
                                        <h3 className="font-black text-sm uppercase tracking-widest text-muted-foreground/60 mb-6 flex items-center gap-2">
                                            <Layers size={14} /> Mission Control
                                        </h3>
                                        <div className="space-y-4">
                                            {assessment.action_items.map((item, i) => (
                                                <div key={i} className="flex gap-3 p-4 bg-secondary/40 border border-border/20 rounded-2xl text-sm font-bold group hover:border-primary/40 transition-colors">
                                                    <div className="w-5 h-5 rounded-full bg-primary/20 flex items-center justify-center shrink-0 group-hover:bg-primary group-hover:text-white transition-colors">
                                                        <ArrowRight size={10} />
                                                    </div>
                                                    {item}
                                                </div>
                                            ))}
                                        </div>
                                    </AppleCard>

                                    <AppleCard className="p-8 border-border/40 rounded-[2.5rem] bg-secondary/30">
                                        <h3 className="font-black text-xs uppercase tracking-widest text-muted-foreground/60 mb-6 flex items-center gap-2">
                                            <PieChart size={14} /> Bullet Distribution
                                        </h3>
                                        <div className="grid grid-cols-2 gap-4">
                                            {Object.entries(assessment.resume_distribution).map(([area, count]) => (
                                                <div key={area} className="p-4 bg-card text-foreground border border-border/40 rounded-2xl text-center shadow-sm">
                                                    <div className="text-2xl font-black text-primary mb-1">{count}</div>
                                                    <div className="text-[10px] font-black uppercase tracking-widest text-muted-foreground/80 truncate">{area.replace("_", " ")}</div>
                                                </div>
                                            ))}
                                        </div>
                                    </AppleCard>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {analysisError && !assessment && !isAnalyzing && (
                    <div role="alert" className="mx-auto max-w-2xl rounded-3xl border border-rose-500/30 bg-rose-500/10 p-6 text-center text-sm font-medium text-foreground">
                        {analysisError}
                    </div>
                )}

                {/* Initial State / Empty Tracker */}
                {!assessment && !isAnalyzing && !analysisError && (
                    <motion.div {...FADE_IN} className="flex flex-col items-center justify-center py-20 opacity-40">
                        <div className="w-24 h-24 rounded-[2rem] border-2 border-dashed border-border/60 flex items-center justify-center mb-6">
                            <BarChart3 size={40} />
                        </div>
                        <h3 className="text-xl font-bold mb-2">No Analysis Loaded</h3>
                        <p className="text-sm font-medium">Connect your Profile and target JD to calculate alignment.</p>
                    </motion.div>
                )}
            </div>
        </div>
    );
}

