"use client";

import { motion, useScroll, useTransform } from "framer-motion";
import Link from "next/link";
import { useRef } from "react";
import {
  Sparkles,
  FileText,
  MessageSquare,
  Target,
  LayoutDashboard,
  ArrowRight,
} from "lucide-react";
import { AppleButton } from "@/components/ui/AppleButton";
import { AppleCard } from "@/components/ui/AppleCard";
import { HeroDashboard } from "@/components/home/HeroDashboard";
import { LogoTicker } from "@/components/home/LogoTicker";
import { Testimonials } from "@/components/home/Testimonials";
import { HowItWorks } from "@/components/home/HowItWorks";

const FloatingOrb = ({
  className,
  delay = 0,
}: {
  className?: string;
  delay?: number;
}) => (
  <motion.div
    className={`absolute rounded-full blur-3xl opacity-30 ${className}`}
    animate={{
      y: [0, -30, 0],
      scale: [1, 1.1, 1],
      opacity: [0.3, 0.5, 0.3],
    }}
    transition={{
      duration: 8,
      delay,
      repeat: Infinity,
      ease: "easeInOut",
    }}
  />
);

export default function Home() {
  const heroRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ["start start", "end start"],
  });
  const heroOpacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);
  const heroScale = useTransform(scrollYProgress, [0, 0.5], [1, 0.95]);
  const dashboardY = useTransform(scrollYProgress, [0, 0.5], [0, 50]);

  return (
    <div className="min-h-screen overflow-hidden bg-background">
      {/* Hero Section */}
      <section
        ref={heroRef}
        className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 overflow-visible"
      >
        {/* Animated Background */}
        <div className="absolute inset-0 bg-gradient-to-b from-primary/5 via-background to-background" />
        <div className="absolute inset-x-0 top-0 h-[500px] bg-gradient-to-b from-blue-500/10 to-transparent opacity-50 blur-3xl" />

        {/* Orbs */}
        <FloatingOrb className="w-[800px] h-[800px] bg-blue-500/20 -top-[400px] left-1/2 -translate-x-1/2" />

        <div className="max-w-[1400px] mx-auto px-6 relative z-10">
          <motion.div
            style={{ opacity: heroOpacity, scale: heroScale }}
            className="flex flex-col items-center text-center max-w-5xl mx-auto"
          >
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/50 dark:bg-slate-900/50 backdrop-blur-md border border-primary/20 shadow-sm mb-8"
            >
              <div className="flex -space-x-2">
                {[1, 2, 3].map((i) => (
                  <div
                    key={i}
                    className="w-6 h-6 rounded-full bg-gradient-to-br from-blue-400 to-purple-500 border-2 border-white dark:border-slate-900"
                  />
                ))}
              </div>
              <span className="text-sm font-medium pl-2">
                Used by engineers from top tech companies
              </span>
            </motion.div>

            {/* Headline */}
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-6xl sm:text-7xl lg:text-8xl font-black tracking-tighter mb-8 leading-[0.9]"
            >
              Land Your Dream Job.
              <br />
              <span className="bg-gradient-to-r from-blue-600 via-indigo-500 to-purple-500 bg-clip-text text-transparent pb-4">
                Powered by AI.
              </span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-xl sm:text-2xl text-muted-foreground max-w-3xl mx-auto mb-12 leading-relaxed"
            >
              Stop guessing keywords. Our AI analyzes job descriptions, tailors
              your resume, and writes your cover letters.{" "}
              <span className="text-foreground font-semibold">Instantly.</span>
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="flex flex-wrap justify-center gap-4 mb-20"
            >
              <Link href="/jobs">
                <AppleButton
                  size="lg"
                  className="h-16 px-8 text-lg font-bold shadow-xl shadow-blue-500/20"
                >
                  Analyze a Job
                  <Sparkles
                    size={18}
                    className="ml-2 group-hover:rotate-12 transition-transform"
                  />
                </AppleButton>
              </Link>
              <Link href="/profile">
                <AppleButton
                  variant="outline"
                  size="lg"
                  className="h-16 px-8 text-lg font-bold border-2"
                >
                  Upload Resume
                </AppleButton>
              </Link>
            </motion.div>
          </motion.div>

          {/* New Hero Dashboard Component */}
          <motion.div
            style={{ y: dashboardY }}
            initial={{ opacity: 0, y: 100, rotateX: 20 }}
            animate={{ opacity: 1, y: 0, rotateX: 0 }}
            transition={{ delay: 0.4, duration: 1, type: "spring" }}
          >
            <HeroDashboard />
          </motion.div>
        </div>
      </section>

      {/* Social Proof */}
      <LogoTicker />

      {/* How It Works */}
      <HowItWorks />

      {/* Bento Grid Features */}
      <section className="py-32 relative bg-slate-50/50 dark:bg-[#0B0F19]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center max-w-3xl mx-auto mb-20">
            <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
              Built for the Modern Job Market
            </h2>
            <p className="text-xl text-muted-foreground">
              Every tool you need to beat the ATS and impress hiring managers.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 md:grid-rows-2 gap-6 h-auto md:h-[800px]">
            {/* 1. Large Feature */}
            <AppleCard className="md:col-span-2 md:row-span-2 rounded-[2.5rem] p-10 relative overflow-hidden group hover:shadow-xl transition-shadow border-slate-200 dark:border-slate-800">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="relative z-10 h-full flex flex-col">
                <div className="w-16 h-16 rounded-2xl bg-blue-100 dark:bg-blue-900/30 text-blue-600 flex items-center justify-center mb-6">
                  <Target size={32} />
                </div>
                <h3 className="text-3xl font-bold mb-4">
                  Precision Job Analysis
                </h3>
                <p className="text-muted-foreground text-lg mb-8 max-w-md">
                  Our AI dissects job descriptions to extract hidden keywords,
                  requirements, and culture signals.
                </p>
                <div className="mt-auto rounded-xl bg-slate-100 dark:bg-slate-800 p-6 border border-slate-200 dark:border-slate-700">
                  <div className="flex items-center gap-4 mb-4">
                    <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                    <span className="text-sm font-mono text-muted-foreground">
                      Analysis Complete
                    </span>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm font-medium">
                      <span>React.js</span>{" "}
                      <span className="text-green-500">Match</span>
                    </div>
                    <div className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                      <div className="w-full h-full bg-green-500" />
                    </div>
                  </div>
                </div>
              </div>
            </AppleCard>

            {/* 2. Top Right */}
            <AppleCard className="rounded-[2.5rem] p-8 relative overflow-hidden group hover:shadow-xl transition-shadow border-slate-200 dark:border-slate-800">
              <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/30 text-purple-600 flex items-center justify-center mb-4">
                <FileText size={24} />
              </div>
              <h3 className="text-xl font-bold mb-2">Resume Optimization</h3>
              <p className="text-muted-foreground">
                Tailor your resume for every single application instantly.
              </p>
            </AppleCard>

            {/* 3. Middle Right */}
            <AppleCard className="rounded-[2.5rem] p-8 relative overflow-hidden group hover:shadow-xl transition-shadow border-slate-200 dark:border-slate-800">
              <div className="w-12 h-12 rounded-xl bg-orange-100 dark:bg-orange-900/30 text-orange-600 flex items-center justify-center mb-4">
                <MessageSquare size={24} />
              </div>
              <h3 className="text-xl font-bold mb-2">Cover Letters</h3>
              <p className="text-muted-foreground">
                Generate compelling cover letters that actually get read.
              </p>
            </AppleCard>

            {/* 4. Bottom Right - Dashboard */}
            <AppleCard className="md:col-span-1 rounded-[2.5rem] bg-gradient-to-br from-[#0F172A] to-[#1E293B] text-white p-8 relative overflow-hidden hover:shadow-2xl transition-shadow border-slate-200 dark:border-slate-800">
              <div className="relative z-10">
                <h3 className="text-2xl font-bold mb-2">Track Everything</h3>
                <p className="text-slate-400 mb-6">
                  Keep all your applications organized.
                </p>
                <Link
                  href="/dashboard"
                  className="text-blue-400 font-medium hover:text-blue-300 flex items-center gap-2"
                >
                  Go to Dashboard <ArrowRight size={16} />
                </Link>
              </div>
              <div className="absolute right-[-20px] bottom-[-20px] opacity-20 rotate-[-15deg]">
                <LayoutDashboard size={120} />
              </div>
            </AppleCard>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <Testimonials />

      {/* CTA Section */}
      <section className="py-32 relative overflow-hidden">
        <div className="absolute inset-0 bg-blue-600" />
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20" />
        <div className="max-w-4xl mx-auto px-6 relative z-10 text-center text-white">
          <h2 className="text-5xl md:text-7xl font-black tracking-tighter mb-8 max-w-2xl mx-auto">
            Ready to accelerate your career?
          </h2>
          <p className="text-xl md:text-2xl text-blue-100 mb-12 max-w-xl mx-auto">
            Join thousands of job seekers using CareerAgentPro to land their
            dream roles.
          </p>
          <Link href="/jobs">
            <button className="px-10 py-5 rounded-full bg-white text-blue-600 font-bold text-xl shadow-2xl hover:bg-blue-50 transition-all hover:scale-105 active:scale-95">
              Get Started for Free
            </button>
          </Link>
        </div>
      </section>
    </div>
  );
}
