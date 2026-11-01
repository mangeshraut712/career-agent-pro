"use client";

import { useState } from "react";
import { Link2, Search, Loader2 } from "lucide-react";
import { motion } from "framer-motion";
import { AppleCard } from "@/components/ui/AppleCard";
import { AppleButton } from "@/components/ui/AppleButton";

interface JobScannerProps {
  onAnalyze: (url: string) => Promise<void>;
  isAnalyzing: boolean;
}

export function JobScanner({ onAnalyze, isAnalyzing }: JobScannerProps) {
  const [url, setUrl] = useState("");

  const handleSubmit = () => {
    if (url) onAnalyze(url);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="max-w-4xl mx-auto mb-20 relative group"
    >
      <div className="absolute inset-0 bg-gradient-to-r from-primary/20 via-purple-500/20 to-pink-500/20 rounded-[2.5rem] blur-3xl opacity-50 group-hover:opacity-75 transition-opacity -z-10" />

      <AppleCard className="p-8 sm:p-10 border-border/40 bg-card/80 backdrop-blur-xl rounded-[2.5rem] shadow-2xl shadow-primary/5">
        <div className="flex flex-col md:flex-row gap-4 items-center">
          <div className="w-full relative">
            <Link2 className="absolute left-5 top-1/2 -translate-y-1/2 w-6 h-6 text-muted-foreground" />
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Paste job URL (LinkedIn, Indeed, Greenhouse...)"
              className="w-full pl-14 pr-6 py-5 bg-secondary/50 rounded-2xl text-base focus:outline-none focus:ring-2 focus:ring-primary/40 border border-border/30 transition-all font-medium"
              onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
            />
          </div>
          <AppleButton
            variant="primary"
            onClick={handleSubmit}
            disabled={isAnalyzing}
            className="w-full md:w-auto px-8 py-5 h-auto text-lg font-bold shadow-xl shadow-primary/25 relative overflow-hidden group"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-primary to-blue-600 opacity-0 group-hover:opacity-100 transition-opacity" />
            <span className="relative z-10 flex items-center gap-2">
              {isAnalyzing ? (
                <>
                  <Loader2 size={20} className="animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Search size={20} />
                  Extract Now
                </>
              )}
            </span>
          </AppleButton>
        </div>
      </AppleCard>
    </motion.div>
  );
}
