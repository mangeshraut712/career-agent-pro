"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { AppleCard } from "@/components/ui/AppleCard";
import { AppleButton } from "@/components/ui/AppleButton";
import { Briefcase, Building2, Plus, ExternalLink, Trash2, Link2 } from "lucide-react";
import { sanitizeUrl } from "@/lib/secureStorage";

interface TrackedJob {
    id: string;
    title: string;
    company: string;
    url: string;
    status: "analyzing" | "resume_updated" | "applied" | "interviewing" | "offer" | "rejected";
    createdAt: string;
}

const STATUS_CONFIG = {
    analyzing: { label: "Analyzing", color: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400", gradient: "from-blue-500 to-cyan-400" },
    resume_updated: { label: "Resume Ready", color: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400", gradient: "from-purple-500 to-pink-400" },
    applied: { label: "Applied", color: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400", gradient: "from-green-500 to-emerald-400" },
    interviewing: { label: "Interviewing", color: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400", gradient: "from-yellow-500 to-orange-400" },
    offer: { label: "Offer", color: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400", gradient: "from-emerald-500 to-green-400" },
    rejected: { label: "Rejected", color: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400", gradient: "from-red-500 to-pink-400" },
};

interface JobTrackerProps {
    jobs: TrackedJob[];
    onUpdateStatus: (id: string, status: TrackedJob["status"]) => void;
    onDelete: (id: string) => void;
}

export function JobTracker({ jobs, onUpdateStatus, onDelete }: JobTrackerProps) {
    return (
        <AppleCard className="overflow-hidden border-border/40 min-h-[500px] flex flex-col">
            {/* Card header with gradient */}
            <div className="bg-gradient-to-r from-primary/5 via-purple-500/5 to-pink-500/5 border-b border-border/40 p-6">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center shadow-lg">
                            <Briefcase className="w-5 h-5 text-white" />
                        </div>
                        <h2 className="text-xl font-semibold">Job Applications</h2>
                    </div>
                    <Link href="/jobs">
                        <AppleButton variant="secondary" size="sm" className="shadow-sm">
                            <Plus size={14} />
                            Add Job
                        </AppleButton>
                    </Link>
                </div>
            </div>

            <div className="p-6 flex-1">
                {jobs.length > 0 ? (
                    <div className="space-y-3">
                        {jobs.map((job) => (
                            <motion.div
                                key={job.id}
                                whileHover={{ x: 4, scale: 1.01 }}
                                className="group relative flex items-center justify-between p-4 rounded-2xl bg-gradient-to-br from-secondary/30 to-secondary/50 hover:from-secondary/50 hover:to-secondary/70 border border-border/20 hover:border-primary/20 transition-all"
                            >
                                {/* Status indicator bar */}
                                <div className={`absolute left-0 top-0 bottom-0 w-1 rounded-l-2xl bg-gradient-to-b ${STATUS_CONFIG[job.status].gradient}`} />

                                <div className="flex items-center gap-4 flex-1 min-w-0">
                                    <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${STATUS_CONFIG[job.status].gradient} flex items-center justify-center font-bold text-white text-lg shadow-md flex-shrink-0`}>
                                        {job.company.charAt(0)}
                                    </div>
                                    <div className="min-w-0 flex-1">
                                        <h4 className="font-semibold text-sm sm:text-base truncate">{job.title}</h4>
                                        <p className="text-sm text-muted-foreground flex items-center gap-1.5 truncate">
                                            <Building2 size={12} className="flex-shrink-0" />
                                            {job.company}
                                        </p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2 flex-shrink-0">
                                    <select
                                        value={job.status}
                                        aria-label={`Application status for ${job.title}`}
                                        onChange={(e) => onUpdateStatus(job.id, e.target.value as TrackedJob["status"])}
                                        className={`text-xs font-semibold px-3 py-1.5 rounded-full border-0 cursor-pointer shadow-sm ${STATUS_CONFIG[job.status].color}`}
                                    >
                                        {Object.entries(STATUS_CONFIG).map(([key, config]) => (
                                            <option key={key} value={key}>{config.label}</option>
                                        ))}
                                    </select>
                                    <a
                                        href={sanitizeUrl(job.url)}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        aria-label={`Open job posting for ${job.title}`}
                                        className="p-2 rounded-lg hover:bg-background/50 transition-colors"
                                    >
                                        <ExternalLink size={14} className="text-muted-foreground hover:text-primary transition-colors" />
                                    </a>
                                    <button
                                        type="button"
                                        onClick={() => onDelete(job.id)}
                                        aria-label={`Remove ${job.title} from tracker`}
                                        className="p-2 rounded-lg hover:bg-background/50 transition-colors text-muted-foreground hover:text-destructive"
                                    >
                                        <Trash2 size={14} />
                                    </button>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                ) : (
                    <div className="text-center py-16 h-full flex flex-col items-center justify-center">
                        <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-primary/10 to-purple-500/10 flex items-center justify-center mx-auto mb-6 border border-primary/20">
                            <Briefcase className="w-10 h-10 text-primary" />
                        </div>
                        <h3 className="font-semibold text-lg mb-2">No jobs tracked yet</h3>
                        <p className="text-sm text-muted-foreground mb-6 max-w-sm mx-auto">
                            Start by analyzing a job posting to track your applications
                        </p>
                        <Link href="/jobs">
                            <AppleButton variant="primary" className="shadow-lg shadow-primary/20">
                                <Link2 size={14} />
                                Analyze Your First Job
                            </AppleButton>
                        </Link>
                    </div>
                )}
            </div>
        </AppleCard>
    );
}
