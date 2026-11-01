"use client";

import { motion } from "framer-motion";
import { AppleCard } from "@/components/ui/AppleCard";
import { LucideIcon } from "lucide-react";

interface Stat {
  label: string;
  value: string;
  icon: LucideIcon;
  gradient: string;
  change: string;
}

interface StatsOverviewProps {
  stats: Stat[];
}

export function StatsOverview({ stats }: StatsOverviewProps) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 mb-8 sm:mb-12">
      {stats.map((stat, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: index * 0.1 }}
          whileHover={{ y: -4, scale: 1.02 }}
          className="group relative"
        >
          <div
            className="absolute inset-0 bg-gradient-to-br opacity-0 group-hover:opacity-100 transition-opacity rounded-2xl blur-xl -z-10"
            style={{
              background: `linear-gradient(to bottom right, var(--tw-gradient-stops))`,
            }}
          />
          <AppleCard className="relative overflow-hidden border-border/40 hover:border-primary/30 transition-all">
            {/* Top gradient bar */}
            <div
              className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${stat.gradient}`}
            />

            <div className="p-5 sm:p-6">
              <div className="flex items-start justify-between mb-4">
                <div
                  className={`w-12 h-12 rounded-xl bg-gradient-to-br ${stat.gradient} flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform`}
                >
                  <stat.icon className="w-6 h-6 text-white" />
                </div>
                <span className="text-xs font-medium text-muted-foreground bg-secondary/50 px-2 py-1 rounded-full">
                  {stat.change}
                </span>
              </div>
              <div className="text-3xl sm:text-4xl font-bold mb-2 bg-gradient-to-br from-foreground to-foreground/70 bg-clip-text text-transparent">
                {stat.value}
              </div>
              <div className="text-sm text-muted-foreground font-medium">
                {stat.label}
              </div>
            </div>
          </AppleCard>
        </motion.div>
      ))}
    </div>
  );
}
