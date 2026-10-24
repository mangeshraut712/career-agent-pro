"use client";

import { useState, useEffect } from "react";
import { FADE_IN } from "@/lib/animations";
import { motion } from "framer-motion";
import Link from "next/link";
import {
    Link2, FileText, MessageSquare, Target,
    Award, Sparkles, Clock
} from "lucide-react";
import { AppleCard } from "@/components/ui/AppleCard";
import { AppleButton } from "@/components/ui/AppleButton";
import { useToast } from "@/components/ui/Toast";
import { secureGet, secureSet } from "@/lib/secureStorage";
import { STORAGE_KEYS } from "@/lib/storageKeys";
import { JobTracker } from "@/components/dashboard/JobTracker";
import {
    JobStatus,
    removeStoredJob,
    StoredAnalyzedJob,
    toTrackedJob,
    updateStoredJobStatus,
    withStableIds,
} from "@/lib/trackedJobs";



const STATUS_CONFIG = {
    analyzing: { label: "Analyzing", color: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400", gradient: "from-blue-500 to-cyan-400" },
    resume_updated: { label: "Resume Ready", color: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400", gradient: "from-purple-500 to-pink-400" },
    applied: { label: "Applied", color: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400", gradient: "from-green-500 to-emerald-400" },
    interviewing: { label: "Interviewing", color: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400", gradient: "from-yellow-500 to-orange-400" },
    offer: { label: "Offer", color: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400", gradient: "from-emerald-500 to-green-400" },
    rejected: { label: "Rejected", color: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400", gradient: "from-red-500 to-pink-400" },
};



export default function DashboardPage() {
    const { toast } = useToast();
    const [storedJobs, setStoredJobs] = useState<StoredAnalyzedJob[]>([]);
    const [profileComplete, setProfileComplete] = useState(0);
    const [greeting, setGreeting] = useState("Welcome");
    const trackedJobs = storedJobs.map((job, index) => toTrackedJob(job, index));

    useEffect(() => {
        const hour = new Date().getHours();
        if (hour < 12) setGreeting("Good morning");
        else if (hour < 18) setGreeting("Good afternoon");
        else setGreeting("Good evening");
    }, []);

    useEffect(() => {
        const loadData = () => {
            try {
                const jobs = secureGet<StoredAnalyzedJob[]>(STORAGE_KEYS.ANALYZED_JOBS);
                if (jobs && Array.isArray(jobs)) {
                    const stable = withStableIds(jobs);
                    setStoredJobs(stable);
                    if (stable.some((job, index) => job !== jobs[index])) {
                        secureSet(STORAGE_KEYS.ANALYZED_JOBS, stable);
                    }
                }

                const data = secureGet<{ name?: string; email?: string; experience?: unknown[]; education?: unknown[]; skills?: unknown[] }>('profile');
                if (data) {
                    let score = 0;
                    if (data.name && data.email) score += 25;
                    if (data.experience?.length && data.experience.length > 0) score += 25;
                    if (data.education?.length && data.education.length > 0) score += 25;
                    if (data.skills?.length && data.skills.length >= 3) score += 25;
                    setProfileComplete(score);
                }
            } catch (error) {
                console.error("Failed to load dashboard data:", error);
            }
        };

        requestAnimationFrame(loadData);
    }, []);

    const updateJobStatus = (jobId: string, status: JobStatus) => {
        const updated = updateStoredJobStatus(storedJobs, jobId, status);
        setStoredJobs(updated);
        secureSet(STORAGE_KEYS.ANALYZED_JOBS, updated);
        const labels = {
            analyzing: "Analyzing",
            resume_updated: "Resume Ready",
            applied: "Applied",
            interviewing: "Interviewing",
            offer: "Offer",
            rejected: "Rejected"
        };
        toast(`Status updated to ${labels[status]}`, "success");
    };

    const deleteTrackedJob = (jobId: string) => {
        const updated = removeStoredJob(storedJobs, jobId);
        setStoredJobs(updated);
        secureSet(STORAGE_KEYS.ANALYZED_JOBS, updated);
        toast("Job removed from tracking", "info");
    };

    const mergedActions = [
        {
            href: "/jobs",
            icon: Link2,
            label: "Analyze Job",
            desc: "Start new analysis",
            stat: `${trackedJobs.length}`,
            statLabel: "Analyzed",
            gradient: "from-blue-500 to-cyan-400",
            highlight: true
        },
        {
            href: "/resumes",
            icon: FileText,
            label: "Resume Studio",
            desc: "Optimize your CV",
            stat: `${trackedJobs.filter(j => j.status !== "analyzing").length}`,
            statLabel: "Optimized",
            gradient: "from-purple-500 to-pink-400"
        },
        {
            href: "/communication",
            icon: MessageSquare,
            label: "Messages",
            desc: "Draft outreach",
            stat: "AI",
            statLabel: "Assistant",
            gradient: "from-green-500 to-emerald-400"
        },
        {
            href: "/profile",
            icon: Target,
            label: "My Profile",
            desc: "Career identity",
            stat: `${profileComplete}%`,
            statLabel: "Complete",
            gradient: "from-orange-500 to-red-400"
        },
    ];

    return (
        <div className="min-h-screen bg-gradient-to-b from-background via-background to-secondary/10">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
                {/* Header with Premium Styling */}
                <motion.div {...FADE_IN} className="mb-10 text-center max-w-2xl mx-auto">
                    <div className="relative inline-block mb-3">
                        <h1 className="text-4xl sm:text-5xl font-bold tracking-tight">
                            {greeting}
                            <motion.span
                                animate={{ rotate: [0, 10, -10, 10, 0] }}
                                transition={{ duration: 1.5, repeat: Infinity, repeatDelay: 5 }}
                                className="inline-block ml-3 origin-bottom-right"
                            >
                                👋
                            </motion.span>
                        </h1>
                    </div>
                    <p className="text-lg text-muted-foreground">
                        You are tracking <span className="font-bold text-primary">{trackedJobs.length} opportunities</span>.
                        <br className="hidden sm:block" /> Ready to make your next move?
                    </p>
                </motion.div>

                {/* Hero Grid: Consolidated Stats & Actions */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                    className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 mb-12"
                >
                    {mergedActions.map((action, index) => (
                        <Link key={index} href={action.href}>
                            <motion.div whileHover={{ y: -5, scale: 1.02 }} whileTap={{ scale: 0.98 }} className="h-full">
                                <AppleCard className={`group relative h-full cursor-pointer overflow-hidden transition-all duration-300 ${action.highlight
                                    ? "border-primary/30 shadow-xl shadow-primary/10 bg-gradient-to-br from-primary/5 to-transparent"
                                    : "border-border/40 hover:border-primary/20"
                                    }`}>
                                    {/* Gradient overlay on hover */}
                                    <div className={`absolute inset-0 bg-gradient-to-br ${action.gradient} opacity-0 group-hover:opacity-[0.05] transition-opacity`} />

                                    <div className="p-5 flex flex-col h-full relative z-10">
                                        <div className="flex justify-between items-start mb-4">
                                            <div className={`w-12 h-12 rounded-2xl bg-gradient-to-br ${action.gradient} flex items-center justify-center shadow-md group-hover:scale-110 group-hover:rotate-3 transition-transform`}>
                                                <action.icon className="w-6 h-6 text-white" />
                                            </div>
                                            {action.stat && (
                                                <div className="text-right">
                                                    <div className={`text-xl font-bold bg-gradient-to-br ${action.gradient} bg-clip-text text-transparent`}>
                                                        {action.stat}
                                                    </div>
                                                    <div className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/70">
                                                        {action.statLabel}
                                                    </div>
                                                </div>
                                            )}
                                        </div>

                                        <div className="mt-auto">
                                            <h3 className="font-bold text-lg leading-tight group-hover:text-primary transition-colors mb-1">{action.label}</h3>
                                            <p className="text-xs font-medium text-muted-foreground">{action.desc}</p>
                                        </div>
                                    </div>

                                    {action.highlight && (
                                        <div className="absolute top-0 right-0 p-3">
                                            <span className="relative flex h-2 w-2">
                                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                                                <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                                            </span>
                                        </div>
                                    )}
                                </AppleCard>
                            </motion.div>
                        </Link>
                    ))}
                </motion.div>

                <div className="grid lg:grid-cols-3 gap-6 sm:gap-8">
                    {/* Main Content - Tracked Jobs */}
                    <div className="lg:col-span-2">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5, delay: 0.3 }}
                        >
                            <JobTracker
                                jobs={trackedJobs}
                                onUpdateStatus={updateJobStatus}
                                onDelete={deleteTrackedJob}
                            />
                        </motion.div>
                    </div>

                    {/* Sidebar */}
                    <div className="space-y-6">
                        {/* AI Tips */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5, delay: 0.4 }}
                        >
                            <AppleCard className="overflow-hidden border-border/40">
                                <div className="bg-gradient-to-br from-primary/10 via-purple-500/10 to-pink-500/10 p-6 border-b border-border/20">
                                    <div className="flex items-center gap-3">
                                        <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center shadow-lg">
                                            <Sparkles className="w-6 h-6 text-white" />
                                        </div>
                                        <div>
                                            <h2 className="font-semibold text-lg">AI Tips</h2>
                                            <p className="text-xs text-muted-foreground">Personalized for you</p>
                                        </div>
                                    </div>
                                </div>
                                <div className="p-6 space-y-3">
                                    {profileComplete < 100 && (
                                        <div className="p-4 rounded-xl bg-gradient-to-br from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 border border-yellow-200/50 dark:border-yellow-800/50">
                                            <p className="text-sm text-yellow-700 dark:text-yellow-400 font-medium">
                                                💡 Complete your profile to get better resume optimization
                                            </p>
                                        </div>
                                    )}
                                    <div className="p-4 rounded-xl bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 border border-blue-200/50 dark:border-blue-800/50">
                                        <p className="text-sm text-blue-700 dark:text-blue-400 font-medium">
                                            🎯 Tailor your resume for each job you apply to for 3x more interviews
                                        </p>
                                    </div>
                                    <div className="p-4 rounded-xl bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 border border-purple-200/50 dark:border-purple-800/50">
                                        <p className="text-sm text-purple-700 dark:text-purple-400 font-medium">
                                            ⚡ Use the bullet library to save time on future applications
                                        </p>
                                    </div>
                                </div>
                            </AppleCard>
                        </motion.div>

                        {/* Profile Progress */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5, delay: 0.5 }}
                        >
                            <AppleCard className="p-6 border-border/40">
                                <div className="flex items-center justify-between mb-6">
                                    <h2 className="font-semibold text-lg flex items-center gap-2">
                                        <Award className="w-5 h-5 text-primary" />
                                        Profile Strength
                                    </h2>
                                    <span className="text-3xl font-bold bg-gradient-to-r from-primary to-purple-500 bg-clip-text text-transparent">{profileComplete}%</span>
                                </div>
                                <div className="h-3 bg-secondary/50 rounded-full overflow-hidden mb-6 border border-border/30">
                                    <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: `${profileComplete}%` }}
                                        transition={{ duration: 1, delay: 0.8, ease: "easeOut" }}
                                        className="h-full bg-gradient-to-r from-primary via-purple-500 to-pink-500 rounded-full relative overflow-hidden"
                                    >
                                        {/* Animated shine effect */}
                                        <div className="absolute inset-0 -translate-x-full animate-[shimmer_2s_infinite] bg-gradient-to-r from-transparent via-white/30 to-transparent" />
                                    </motion.div>
                                </div>
                                <Link href="/profile">
                                    <AppleButton variant="secondary" className="w-full shadow-sm">
                                        {profileComplete < 100 ? "Complete Profile" : "View Profile"}
                                    </AppleButton>
                                </Link>
                            </AppleCard>
                        </motion.div>

                        {/* Recent Activity */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5, delay: 0.6 }}
                        >
                            <AppleCard className="p-6 border-border/40">
                                <div className="flex items-center gap-2 mb-6">
                                    <Clock className="w-5 h-5 text-muted-foreground" />
                                    <h2 className="font-semibold text-lg">Recent Activity</h2>
                                </div>
                                <div className="space-y-3">
                                    {trackedJobs.slice(0, 3).map((job) => (
                                        <div key={job.id} className="flex items-center gap-3 text-sm p-3 rounded-xl bg-secondary/30 hover:bg-secondary/50 transition-colors"
                                            title={`${STATUS_CONFIG[job.status] ? "" : "Unknown status"}`}>
                                            <div className={`w-2 h-2 rounded-full bg-gradient-to-r ${STATUS_CONFIG[job.status]?.gradient || "from-gray-500 to-gray-400"} shadow-sm`} />
                                            <span className="text-muted-foreground truncate flex-1">
                                                Analyzed <span className="font-medium text-foreground">{job.title}</span> at {job.company}
                                            </span>
                                        </div>
                                    ))}
                                    {trackedJobs.length === 0 && (
                                        <p className="text-sm text-muted-foreground text-center py-8 bg-secondary/30 rounded-xl">
                                            No activity yet
                                        </p>
                                    )}
                                </div>
                            </AppleCard>
                        </motion.div>
                    </div>
                </div>
            </div>
        </div>
    );
}
