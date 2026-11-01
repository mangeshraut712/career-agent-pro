"use client";

import { motion, HTMLMotionProps } from "framer-motion";
import { cn } from "@/lib/utils";

interface AppleCardProps extends HTMLMotionProps<"div"> {
  children: React.ReactNode;
  className?: string;
  noPadding?: boolean;
}

export function AppleCard({
  children,
  className,
  noPadding = false,
  ...props
}: AppleCardProps) {
  return (
    <motion.div
      className={cn(
        "apple-card bg-card text-card-foreground",
        !noPadding && "p-6",
        className,
      )}
      {...props}
    >
      {children}
    </motion.div>
  );
}
