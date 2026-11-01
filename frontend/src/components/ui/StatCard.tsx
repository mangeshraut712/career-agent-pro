"use client";

import { motion } from "framer-motion";
import { LucideIcon } from "lucide-react";

interface StatCardProps {
  icon: LucideIcon;
  value: string | number;
  label: string;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  color?: string;
}

export function StatCard({
  icon: Icon,
  value,
  label,
  trend,
  color = "primary",
}: StatCardProps) {
  const colorClasses: Record<
    string,
    { bg: string; text: string; shadow: string }
  > = {
    primary: {
      bg: "bg-primary/10",
      text: "text-primary",
      shadow: "shadow-primary/20",
    },
    green: {
      bg: "bg-green-500/10",
      text: "text-green-500",
      shadow: "shadow-green-500/20",
    },
    purple: {
      bg: "bg-purple-500/10",
      text: "text-purple-500",
      shadow: "shadow-purple-500/20",
    },
    orange: {
      bg: "bg-orange-500/10",
      text: "text-orange-500",
      shadow: "shadow-orange-500/20",
    },
  };

  const colors = colorClasses[color] || colorClasses.primary;

  return (
    <motion.div
      whileHover={{ y: -4, scale: 1.02 }}
      transition={{ type: "spring", stiffness: 300, damping: 20 }}
      className={`apple-card p-6 cursor-default ${colors.shadow}`}
    >
      <div className="flex items-start justify-between mb-4">
        <div
          className={`w-12 h-12 rounded-2xl ${colors.bg} flex items-center justify-center`}
        >
          <Icon className={`w-6 h-6 ${colors.text}`} />
        </div>
        {trend && (
          <span
            className={`text-xs font-medium px-2 py-1 rounded-full ${
              trend.isPositive
                ? "bg-green-500/10 text-green-500"
                : "bg-red-500/10 text-red-500"
            }`}
          >
            {trend.isPositive ? "+" : ""}
            {trend.value}%
          </span>
        )}
      </div>
      <motion.h3
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-3xl font-bold tracking-tight mb-1"
      >
        {value}
      </motion.h3>
      <p className="text-sm text-muted-foreground">{label}</p>
    </motion.div>
  );
}

interface StatsGridProps {
  stats: StatCardProps[];
  columns?: 2 | 3 | 4;
}

export function StatsGrid({ stats, columns = 4 }: StatsGridProps) {
  const colClasses = {
    2: "grid-cols-1 sm:grid-cols-2",
    3: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3",
    4: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-4",
  };

  return (
    <div className={`grid ${colClasses[columns]} gap-4`}>
      {stats.map((stat, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          <StatCard {...stat} />
        </motion.div>
      ))}
    </div>
  );
}
