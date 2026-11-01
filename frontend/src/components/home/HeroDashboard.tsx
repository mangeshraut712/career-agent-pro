"use client";

import { motion } from "framer-motion";
import { useEffect, useState } from "react";

export function HeroDashboard() {
  const [matchScore, setMatchScore] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setMatchScore(94);
    }, 500);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="relative mx-auto mt-10 perspective-[2000px]">
      <div className="relative rounded-[2rem] bg-slate-900/5 dark:bg-white/5 p-4 backdrop-blur-3xl border border-white/20 dark:border-white/10 shadow-2xl">
        <div className="absolute inset-0 bg-gradient-to-tr from-blue-500/10 to-purple-500/10 rounded-[2rem]" />
        {/* Fake UI Content */}
        <div className="relative aspect-[16/9] rounded-2xl overflow-hidden bg-white dark:bg-[#0F172A] border border-border/50 shadow-inner flex flex-col">
          {/* Fake Header */}
          <div className="h-16 border-b border-border/40 flex items-center px-8 justify-between">
            <div className="flex gap-4">
              <div className="w-3 h-3 rounded-full bg-red-400" />
              <div className="w-3 h-3 rounded-full bg-yellow-400" />
              <div className="w-3 h-3 rounded-full bg-green-400" />
            </div>
            <div className="h-2 w-32 bg-slate-200 dark:bg-slate-700 rounded-full" />
          </div>
          {/* Fake Body */}
          <div className="flex-1 p-8 grid grid-cols-12 gap-8 bg-slate-50/50 dark:bg-[#0F172A]">
            {/* Sidebar */}
            <div className="col-span-3 space-y-4">
              <div className="h-32 rounded-2xl bg-white dark:bg-slate-800 shadow-sm border border-slate-100 dark:border-slate-700/50 p-4">
                <div className="w-10 h-10 rounded-lg bg-blue-100 dark:bg-blue-900/30 mb-3" />
                <div className="h-2 w-20 bg-slate-200 dark:bg-slate-700 rounded-full mb-2" />
                <div className="h-2 w-12 bg-slate-100 dark:bg-slate-800 rounded-full" />
              </div>
              <div className="h-full rounded-2xl bg-white dark:bg-slate-800 shadow-sm border border-slate-100 dark:border-slate-700/50" />
            </div>
            {/* Main */}
            <div className="col-span-9 space-y-6">
              <div className="grid grid-cols-3 gap-6">
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  className="h-24 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 text-white p-4"
                >
                  <div className="opacity-50 text-xs font-bold uppercase tracking-wider mb-2">
                    Match Score
                  </div>
                  <div className="text-3xl font-black">
                    <Counter value={matchScore} />%
                  </div>
                </motion.div>
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  className="h-24 rounded-2xl bg-white dark:bg-slate-800 shadow-sm border border-slate-100 dark:border-slate-700/50 p-4"
                >
                  <div className="text-muted-foreground text-xs font-bold uppercase tracking-wider mb-2">
                    Skills Found
                  </div>
                  <div className="text-3xl font-black text-foreground">
                    18/20
                  </div>
                </motion.div>
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  className="h-24 rounded-2xl bg-white dark:bg-slate-800 shadow-sm border border-slate-100 dark:border-slate-700/50 p-4"
                >
                  <div className="text-muted-foreground text-xs font-bold uppercase tracking-wider mb-2">
                    Missing
                  </div>
                  <div className="text-3xl font-black text-rose-500">2</div>
                </motion.div>
              </div>
              <div className="h-64 rounded-2xl bg-white dark:bg-slate-800 shadow-sm border border-slate-100 dark:border-slate-700/50 p-6 relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-blue-500/5 to-transparent animate-shimmer" />
                <div className="flex gap-2 mb-4 relative z-10">
                  <div className="h-8 w-24 bg-slate-100 dark:bg-slate-700/50 rounded-lg" />
                  <div className="h-8 w-24 bg-slate-100 dark:bg-slate-700/50 rounded-lg" />
                </div>
                <div className="space-y-3 relative z-10">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: "75%" }}
                    transition={{ duration: 1, delay: 0.5 }}
                    className="h-4 bg-slate-100 dark:bg-slate-700/30 rounded-full"
                  />
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: "50%" }}
                    transition={{ duration: 1, delay: 0.7 }}
                    className="h-4 bg-slate-100 dark:bg-slate-700/30 rounded-full"
                  />
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: "83%" }}
                    transition={{ duration: 1, delay: 0.9 }}
                    className="h-4 bg-slate-100 dark:bg-slate-700/30 rounded-full"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Counter({ value }: { value: number }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    const duration = 1500;
    const steps = 60;
    const stepTime = duration / steps;
    let current = 0;

    const timer = setInterval(() => {
      current += value / steps;
      if (current >= value) {
        setCount(value);
        clearInterval(timer);
      } else {
        setCount(Math.floor(current));
      }
    }, stepTime);

    return () => clearInterval(timer);
  }, [value]);

  return <span>{count}</span>;
}
