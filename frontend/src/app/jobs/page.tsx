"use client";

import { useState, useEffect } from "react";
import { FADE_IN } from "@/lib/animations";
import { motion, AnimatePresence } from "framer-motion";
import {
    Sparkles, History, Globe, Users, TrendingUp,
    Trash2, ArrowRight
} from "lucide-react";
import { AppleCard } from "@/components/ui/AppleCard";
import { useToast } from "@/components/ui/Toast";
import axios from "axios";
import API_URL from "@/lib/api";
import Link from "next/link";
import { secureGet, secureSet } from "@/lib/secureStorage";
import { STORAGE_KEYS } from "@/lib/storageKeys";
import { JobScanner } from "@/components/jobs/JobScanner";
import { AnalysisResult } from "@/components/jobs/AnalysisResult";
import { calculateMatchScore } from "@/lib/matchScore";

interface AnalyzedJob {
    id: string;
    url: string;
    title: string;
    company: string;
    location: string;
    salary: string;
    jobType: string;
    workArrangement?: string;
    experienceLevel?: string;
    yearsExperience?: string;
    aboutJob?: string;
    responsibilities: string[];
    minimumQualifications: string[];
    preferredQualifications: string[];
    aboutCompany?: string;
    whyJoin?: string;
    teamSize?: string;
    fundingInfo?: string;
    description: string;
    requirements: string[];
    skills: string[];
    benefits: string[];
    jobInfo?: Record<string, string>;
    source?: string;
    note?: string;
    analyzedAt: string;
    matchScore?: number;
}

interface UserProfile {
    skills: string[];
}



export default function AnalyzeJobPage() {
    const { toast } = useToast();
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [currentJob, setCurrentJob] = useState<AnalyzedJob | null>(null);
    const [savedJobs, setSavedJobs] = useState<AnalyzedJob[]>([]);
    const [profile, setProfile] = useState<UserProfile | null>(null);

    useEffect(() => {
        const loadData = () => {
            try {
                const jobs = secureGet<AnalyzedJob[]>(STORAGE_KEYS.ANALYZED_JOBS);
                if (jobs) setSavedJobs(jobs);

                const p = secureGet<UserProfile>(STORAGE_KEYS.PROFILE);
                if (p) setProfile(p);
            } catch (error) {
                console.error('Failed to load storage data:', error);
            }
        };
        loadData();
    }, []);

    const handleAnalyze = async (jobUrl: string) => {
        if (!jobUrl.trim() || !jobUrl.startsWith('http')) {
            toast("Please enter a valid job URL", "error");
            return;
        }

        setIsAnalyzing(true);
        try {
            const response = await axios.post(`${API_URL}/extract-job`, { url: jobUrl });
            const data = response.data;

            const analyzedJob: AnalyzedJob = {
                id: Date.now().toString(),
                url: jobUrl,
                title: data.title || "Job Title",
                company: data.company || "Company",
                location: data.location || "Location not specified",
                salary: data.salary_range || "Not specified",
                jobType: data.job_type || "Full-time",
                workArrangement: data.work_arrangement,
                experienceLevel: data.experience_level,
                yearsExperience: data.years_experience,
                aboutJob: data.about_job,
                responsibilities: data.responsibilities || [],
                minimumQualifications: data.minimum_qualifications || [],
                preferredQualifications: data.preferred_qualifications || [],
                aboutCompany: data.about_company,
                whyJoin: data.why_join,
                teamSize: data.team_size,
                fundingInfo: data.funding_info,
                description: data.description || data.about_job || "",
                requirements: data.requirements || data.minimum_qualifications || [],
                skills: data.skills || [],
                benefits: data.benefits || [],
                jobInfo: data.job_info || {},
                source: data.source,
                note: data.note,
                analyzedAt: new Date().toISOString(),
            };

            if (profile?.skills?.length) {
                analyzedJob.matchScore = calculateMatchScore(analyzedJob.skills, profile.skills);
            }

            setCurrentJob(analyzedJob);
            toast("AI Extraction Complete!", "success");
        } catch (error: unknown) {
            console.error("Analysis error:", error);
            if (axios.isAxiosError(error)) {
                if (error.response?.status === 400) {
                    const detail = error.response?.data?.detail || error.response?.data?.error;
                    if (detail?.includes("not found") || detail?.includes("removed")) {
                        toast("Job posting not found. It may have expired or been removed.", "error");
                    } else {
                        toast("Invalid URL or job page format. Try a different link.", "error");
                    }
                } else if (error.response?.status === 500) {
                    toast("Server error. The job site may be blocking our request.", "info");
                } else if (!error.response) {
                    toast("Cannot connect to server. Please check if backend is running.", "error");
                } else {
                    toast("Failed to analyze job. Try pasting the job description directly.", "error");
                }
            } else {
                toast("An unexpected error occurred. Please try again.", "error");
            }
        } finally {
            setIsAnalyzing(false);
        }
    };

    const saveCurrentJob = () => {
        if (!currentJob) return;
        const updatedJobs = [currentJob, ...savedJobs.filter(j => j.url !== currentJob.url)];
        setSavedJobs(updatedJobs);
        secureSet(STORAGE_KEYS.ANALYZED_JOBS, updatedJobs);
        secureSet(STORAGE_KEYS.CURRENT_JOB_FOR_RESUME, currentJob);
        toast("Job saved to your tracker", "success");
    };

    const deleteJob = (jobId: string) => {
        const updatedJobs = savedJobs.filter(j => j.id !== jobId);
        setSavedJobs(updatedJobs);
        secureSet(STORAGE_KEYS.ANALYZED_JOBS, updatedJobs);
        if (currentJob?.id === jobId) setCurrentJob(null);
        toast("Job removed", "info");
    };

    return (
        <div className="min-h-screen bg-gradient-to-b from-background via-background to-secondary/10">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 py-12 lg:py-20">

                {/* Hero Header Section */}
                <motion.div {...FADE_IN} className="text-center mb-16">
                    <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ duration: 0.5 }}
                        className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm font-medium mb-6"
                    >
                        <Sparkles size={16} />
                        <span>AI-Powered Job Analysis</span>
                    </motion.div>
                    <h1 className="text-5xl sm:text-6xl font-bold tracking-tight mb-6">
                        Paste. Analyze. <span className="text-primary italic">Apply.</span>
                    </h1>
                    <p className="text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
                        Instantly extract requirements, skills, and match probability from any job posting.
                    </p>
                </motion.div>

                <JobScanner onAnalyze={handleAnalyze} isAnalyzing={isAnalyzing} />

                {/* Content Grid */}
                <div className="grid lg:grid-cols-12 gap-10 items-start">

                    {/* Left/Main Column: Results */}
                    <div className="lg:col-span-8 space-y-8">
                        <AnimatePresence mode="wait">
                            {currentJob ? (
                                <AnalysisResult key="results" job={currentJob} onSave={saveCurrentJob} />
                            ) : (
                                <motion.div
                                    key="empty"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    className="flex flex-col items-center justify-center py-20 text-center"
                                >
                                    <div className="w-24 h-24 rounded-[2rem] bg-gradient-to-br from-secondary to-secondary flex items-center justify-center mb-8 text-muted-foreground/20 border-2 border-dashed border-border/50">
                                        <History size={48} />
                                    </div>
                                    <h3 className="text-2xl font-bold mb-3 text-foreground/40">No Analysis Active</h3>
                                    <p className="text-muted-foreground max-w-sm mb-10">Paste a job posting URL above to instantly extract requirements and fit score.</p>

                                    <div className="grid grid-cols-2 gap-4 w-full max-w-sm">
                                        <div className="p-4 rounded-3xl bg-card border border-border/40 text-left">
                                            <Globe size={20} className="text-blue-500 mb-3" />
                                            <div className="font-bold text-sm">Any Board</div>
                                            <div className="text-xs text-muted-foreground">LinkedIn, Indeed...</div>
                                        </div>
                                        <div className="p-4 rounded-3xl bg-card border border-border/40 text-left">
                                            <Users size={20} className="text-purple-500 mb-3" />
                                            <div className="font-bold text-sm">Rich Data</div>
                                            <div className="text-xs text-muted-foreground">Team & Funding info</div>
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                    {/* Right Column: Tracker & Tips */}
                    <div className="lg:col-span-4 space-y-8">

                        {/* Saved Jobs Tracker */}
                        <motion.div {...FADE_IN} transition={{ delay: 0.3 }}>
                            <AppleCard className="p-8 border-border/40 rounded-[2rem]">
                                <div className="flex items-center justify-between mb-8">
                                    <div className="flex items-center gap-3">
                                        <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center text-amber-600">
                                            <TrendingUp size={20} />
                                        </div>
                                        <h2 className="text-lg font-bold tracking-tight">Saved Insights</h2>
                                    </div>
                                    <span className="bg-secondary px-2.5 py-1 rounded-lg text-[11px] font-bold text-muted-foreground uppercase tracking-widest">
                                        {savedJobs.length} Items
                                    </span>
                                </div>

                                <div className="space-y-4">
                                    {savedJobs.length > 0 ? (
                                        savedJobs.slice(0, 5).map((job) => (
                                            <motion.div
                                                key={job.id}
                                                whileHover={{ x: 4, y: -2 }}
                                                className={`group flex items-center gap-2 p-4 rounded-2xl transition-all border ${currentJob?.id === job.id
                                                    ? "bg-primary/[0.03] border-primary/20 ring-1 ring-primary/10"
                                                    : "bg-secondary/40 border-border/20 hover:border-primary/20"
                                                    }`}
                                            >
                                                <button
                                                    type="button"
                                                    onClick={() => { setCurrentJob(job); }}
                                                    className="flex flex-1 items-center gap-3 min-w-0 text-left rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                                                >
                                                    <div className="w-10 h-10 rounded-xl bg-background border border-border/50 flex items-center justify-center font-bold text-sm shadow-sm group-hover:scale-110 transition-transform">
                                                        {(job.company || "?").charAt(0)}
                                                    </div>
                                                    <div className="flex-1 min-w-0">
                                                        <h4 className="font-bold text-sm truncate leading-tight group-hover:text-primary transition-colors">{job.title}</h4>
                                                        <p className="text-xs text-muted-foreground truncate">{job.company}</p>
                                                    </div>
                                                </button>
                                                <button
                                                    type="button"
                                                    aria-label={`Remove ${job.title || "saved job"}`}
                                                    onClick={() => deleteJob(job.id)}
                                                    className="p-1.5 rounded-lg text-muted-foreground hover:bg-rose-500/10 hover:text-rose-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500"
                                                >
                                                    <Trash2 size={14} aria-hidden="true" />
                                                </button>
                                            </motion.div>
                                        ))
                                    ) : (
                                        <div className="text-center py-10 px-4 rounded-3xl border border-dashed border-border/50 text-muted-foreground/60 text-sm font-medium">
                                            Analyses will appear here
                                        </div>
                                    )}
                                </div>

                                {savedJobs.length > 0 && (
                                    <Link href="/dashboard" className="block mt-6 text-center text-xs font-bold text-primary hover:underline flex items-center justify-center gap-1 group">
                                        View Complete Tracker <ArrowRight size={12} className="group-hover:translate-x-1 transition-transform" />
                                    </Link>
                                )}
                            </AppleCard>
                        </motion.div>

                        {/* Pro Tips / Wisdom Cards */}
                        <motion.div {...FADE_IN} transition={{ delay: 0.4 }} className="space-y-4">
                            <AppleCard className="p-6 border-border/40 bg-gradient-to-br from-indigo-500 to-purple-600 text-white rounded-[2rem]">
                                <Sparkles size={24} className="mb-4 text-indigo-200" />
                                <h3 className="text-lg font-bold mb-2">Power Move</h3>
                                <p className="text-sm text-indigo-50/80 leading-relaxed font-medium">
                                    Analyze <span className="text-white font-bold">3+ jobs</span> in the same niche to see common skill patterns AI looks for.
                                </p>
                            </AppleCard>
                        </motion.div>
                    </div>
                </div>
            </div>
        </div>
    );
}
