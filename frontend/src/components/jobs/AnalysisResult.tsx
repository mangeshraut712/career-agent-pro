"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import {
    Building2, MapPin, Bookmark, ExternalLink, Target,
    DollarSign, Briefcase, Clock, FileText, ArrowRight,
    CheckCircle2, Zap, Sparkles, ShieldCheck
} from "lucide-react";
import { AppleCard } from "@/components/ui/AppleCard";
import { AppleButton } from "@/components/ui/AppleButton";

interface AnalyzedJob {
    id: string;
    url: string;
    title: string;
    company: string;
    location: string;
    salary: string;
    jobType: string;
    yearsExperience?: string;
    description: string;
    requirements: string[];
    minimumQualifications: string[];
    skills: string[];
    matchScore?: number;
}

interface AnalysisResultProps {
    job: AnalyzedJob;
    onSave: () => void;
}

function SafeExternalIconLink({ url }: { url: string }) {
    let href: string | undefined;
    try {
        const parsed = new URL(url.trim());
        const rest = `${parsed.host}${parsed.pathname}${parsed.search}${parsed.hash}`;
        // Literal scheme prefix: javascript:/data: cannot survive this rebuild.
        if (parsed.protocol === "https:") {
            href = `https://${rest}`;
        } else if (parsed.protocol === "http:") {
            href = `http://${rest}`;
        }
    } catch {
        href = undefined;
    }

    if (href === undefined) {
        return null;
    }

    const safeHref = href;
    const openOriginalPosting = () => {
        if (safeHref.startsWith("https://") || safeHref.startsWith("http://")) {
            window.open(safeHref, "_blank", "noopener,noreferrer");
        }
    };

    return (
        <button
            type="button"
            onClick={openOriginalPosting}
            aria-label="Open original job posting"
            className="flex items-center justify-center bg-white/50 backdrop-blur-sm border border-border/50 hover:bg-white shadow-sm h-12 w-12 p-0 rounded-full transition-all hover:scale-105 active:scale-95"
        >
            <ExternalLink size={20} className="text-muted-foreground" />
        </button>
    );
}

export function AnalysisResult({ job, onSave }: AnalysisResultProps) {
    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4 }}
        >
            <AppleCard className="overflow-hidden border-border/40 shadow-xl rounded-[2rem]">
                {/* Header Branding Row */}
                <div className="p-8 sm:p-10 border-b border-border/40 bg-gradient-to-br from-primary/[0.03] to-purple-500/[0.03]">
                    <div className="flex flex-col sm:flex-row items-start justify-between gap-6">
                        <div className="flex items-start gap-5">
                            <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center font-bold text-3xl text-white shadow-2xl shadow-primary/20 shrink-0">
                                {job.company.charAt(0)}
                            </div>
                            <div>
                                <h2 className="text-3xl font-bold tracking-tight mb-2 leading-none">{job.title}</h2>
                                <div className="flex flex-wrap items-center gap-4 text-muted-foreground font-medium">
                                    <div className="flex items-center gap-1.5 bg-secondary/70 px-3 py-1 rounded-full text-sm">
                                        <Building2 size={16} />
                                        {job.company}
                                    </div>
                                    <div className="flex items-center gap-1.5 bg-secondary/70 px-3 py-1 rounded-full text-sm">
                                        <MapPin size={16} />
                                        {job.location}
                                    </div>
                                </div>
                            </div>
                        </div>


                        <div className="flex gap-2">
                            <AppleButton onClick={onSave} variant="secondary" className="bg-white/50 backdrop-blur-sm border border-border/50 hover:bg-white shadow-sm h-12 w-12 p-0 flex items-center justify-center">
                                <Bookmark size={20} className="text-primary" />
                            </AppleButton>
                            <SafeExternalIconLink url={job.url} />
                        </div>
                    </div>

                    {/* Key Specs Pills */}
                    <div className="flex flex-wrap gap-3 mt-8">
                        {job.matchScore !== undefined && (
                            <div className={`flex items-center gap-2 px-5 py-2.5 rounded-2xl font-bold text-base shadow-sm ${job.matchScore > 75 ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400" :
                                job.matchScore > 45 ? "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400" :
                                    "bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400"
                                }`}>
                                <Target size={18} />
                                {job.matchScore}% Skill Match
                            </div>
                        )}
                        <div className="flex items-center gap-2 px-5 py-2.5 rounded-2xl bg-secondary/80 text-foreground font-semibold shadow-sm text-sm border border-border/20">
                            <DollarSign size={16} className="text-emerald-500" />
                            {job.salary}
                        </div>
                        <div className="flex items-center gap-2 px-5 py-2.5 rounded-2xl bg-secondary/80 text-foreground font-semibold shadow-sm text-sm border border-border/20">
                            <Briefcase size={16} className="text-primary" />
                            {job.jobType}
                        </div>
                        {job.yearsExperience && (
                            <div className="flex items-center gap-2 px-5 py-2.5 rounded-2xl bg-secondary/80 text-foreground font-semibold shadow-sm text-sm border border-border/20">
                                <Clock size={16} className="text-purple-500" />
                                {job.yearsExperience}
                            </div>
                        )}
                    </div>
                </div>

                {/* Structured Content Sections */}
                <div className="p-8 sm:p-10 space-y-12">

                    {/* About Job */}
                    {job.description && (
                        <section>
                            <div className="flex items-center gap-2 mb-4">
                                <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
                                    <FileText size={18} />
                                </div>
                                <h3 className="text-lg font-bold">Role Intelligence</h3>
                            </div>
                            <div className="p-6 rounded-[1.5rem] bg-secondary/30 border border-border/20 text-foreground/80 leading-relaxed text-base">
                                {job.description.split('\n').map((line, i) => (
                                    <p key={i} className={line.trim() ? "mb-4" : "h-2"}>{line}</p>
                                )).slice(0, 4)}
                                <p className="text-primary font-semibold text-sm cursor-pointer hover:underline inline-flex items-center gap-1">
                                    View full description <ArrowRight size={14} />
                                </p>
                            </div>
                        </section>
                    )}

                    <div className="grid md:grid-cols-2 gap-8">
                        {/* Requirements */}
                        <section>
                            <div className="flex items-center gap-2 mb-5 font-bold text-base uppercase tracking-widest text-muted-foreground/80">
                                <Target size={16} className="text-rose-500" />
                                Essential Requirements
                            </div>
                            <div className="space-y-3">
                                {(job.requirements || job.minimumQualifications).slice(0, 6).map((req, i) => (
                                    <div key={i} className="flex gap-3 p-4 rounded-xl bg-card border border-border/40 shadow-sm text-sm leading-snug">
                                        <CheckCircle2 size={18} className="text-emerald-500 shrink-0 mt-0.5" />
                                        {req}
                                    </div>
                                ))}
                            </div>
                        </section>

                        {/* Skills Mesh */}
                        <section>
                            <div className="flex items-center gap-2 mb-5 font-bold text-base uppercase tracking-widest text-muted-foreground/80">
                                <Zap size={16} className="text-amber-500" />
                                Critical Skills
                            </div>
                            <div className="flex flex-wrap gap-2.5">
                                {job.skills.map((skill, i) => (
                                    <motion.div
                                        key={i}
                                        whileHover={{ scale: 1.05, y: -2 }}
                                        className="px-4 py-2.5 rounded-xl bg-gradient-to-br from-primary/5 to-primary/10 border border-primary/20 text-primary font-bold text-sm shadow-sm"
                                    >
                                        {skill}
                                    </motion.div>
                                ))}
                            </div>
                        </section>
                    </div>

                    {/* CTA Multi-Action Bar */}
                    <div className="pt-6 border-t border-border/40">
                        <div className="flex flex-wrap gap-4">
                            <Link href="/resumes" className="flex-1 min-w-[200px]" onClick={onSave}>
                                <AppleButton className="w-full h-14 bg-gradient-to-r from-primary to-purple-600 font-bold tracking-tight text-white gap-2 shadow-xl shadow-primary/20">
                                    <Sparkles size={18} />
                                    Optimize Resume
                                </AppleButton>
                            </Link>
                            <Link href="/outreach" className="flex-1 min-w-[200px]" onClick={onSave}>
                                <AppleButton variant="secondary" className="w-full h-14 font-bold border-border/60 hover:border-primary/40 bg-white/50 backdrop-blur-sm gap-2">
                                    <Target size={18} className="text-primary" />
                                    Plan Strategy
                                </AppleButton>
                            </Link>
                            <Link href="/fit-analysis" className="flex-1 min-w-[200px]" onClick={onSave}>
                                <AppleButton variant="ghost" className="w-full h-14 font-bold border border-border/30 hover:bg-secondary/50 gap-2">
                                    <ShieldCheck size={18} className="text-muted-foreground" />
                                    Deep Fit Check
                                </AppleButton>
                            </Link>
                        </div>
                    </div>
                </div>
            </AppleCard>
        </motion.div>
    );
}
