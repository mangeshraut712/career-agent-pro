"use client";

import { motion } from "framer-motion";

const companies = [
  "Google",
  "Microsoft",
  "Meta",
  "Amazon",
  "Netflix",
  "Apple",
  "Uber",
  "Airbnb",
  "Spotify",
  "Stripe",
];

export function LogoTicker() {
  return (
    <section className="py-10 bg-slate-50 dark:bg-slate-900/50 border-y border-border/40 overflow-hidden">
      <div className="max-w-7xl mx-auto px-6 mb-8 text-center">
        <p className="text-sm font-semibold text-muted-foreground uppercase tracking-widest">
          Trusted by engineers at
        </p>
      </div>
      <div className="relative flex overflow-x-hidden">
        <div className="absolute left-0 top-0 bottom-0 w-32 bg-gradient-to-r from-slate-50 dark:from-[#0B0F19] to-transparent z-10" />
        <div className="absolute right-0 top-0 bottom-0 w-32 bg-gradient-to-l from-slate-50 dark:from-[#0B0F19] to-transparent z-10" />

        <motion.div
          animate={{ x: "-50%" }}
          transition={{
            duration: 20,
            ease: "linear",
            repeat: Infinity,
          }}
          className="flex gap-16 whitespace-nowrap px-8"
        >
          {[...companies, ...companies].map((company, index) => (
            <div
              key={index}
              className="text-xl md:text-2xl font-black text-muted-foreground/50 hover:text-foreground transition-colors cursor-default"
            >
              {company}
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
