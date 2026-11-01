"use client";

import { motion } from "framer-motion";
import { AppleCard } from "@/components/ui/AppleCard";
import { Star } from "lucide-react";

const testimonials = [
  {
    name: "Alex C.",
    role: "Senior Frontend Engineer",
    text: "CareerAgentPro completely changed my job search. I went from 0 callbacks to 5 interviews in one week.",
    company: "TechCorp",
  },
  {
    name: "Sarah L.",
    role: "Product Manager",
    text: "The resume optimization is magic. It picked up keywords I completely missed.",
    company: "StartUp Inc",
  },
  {
    name: "Michael R.",
    role: "Backend Dev",
    text: "Finally, a tool that actually helps with the ATS black hole. Highly recommend.",
    company: "Global Systems",
  },
  {
    name: "Emily W.",
    role: "UX Designer",
    text: "The cover letter generator saves me hours. I can finally customize every application.",
    company: "Design Studio",
  },
  {
    name: "David K.",
    role: "Data Scientist",
    text: "Fit Analysis gave me the confidence to apply for reach roles. And I got one!",
    company: "AI Labs",
  },
];

export function Testimonials() {
  return (
    <section className="py-24 overflow-hidden bg-background relative">
      <div className="max-w-7xl mx-auto px-6 mb-16 text-center">
        <h2 className="text-4xl font-bold tracking-tight mb-4">
          Don&apos;t just take our word for it.
        </h2>
        <p className="text-xl text-muted-foreground">
          Join thousands of professionals accelerating their careers.
        </p>
      </div>

      <div className="relative flex overflow-x-hidden group">
        <div className="absolute left-0 top-0 bottom-0 w-32 bg-gradient-to-r from-background to-transparent z-10" />
        <div className="absolute right-0 top-0 bottom-0 w-32 bg-gradient-to-l from-background to-transparent z-10" />

        <motion.div
          animate={{ x: "-50%" }}
          transition={{
            duration: 40,
            ease: "linear",
            repeat: Infinity,
          }}
          className="flex gap-6 px-6 w-max"
        >
          {[...testimonials, ...testimonials].map((t, i) => (
            <AppleCard
              key={i}
              className="w-[350px] p-6 flex-shrink-0 border-border/50 hover:border-primary/30 transition-colors bg-secondary/20"
            >
              <div className="flex gap-1 text-amber-400 mb-4">
                {[1, 2, 3, 4, 5].map((s) => (
                  <Star key={s} size={16} fill="currentColor" />
                ))}
              </div>
              <p className="text-lg font-medium leading-relaxed mb-6">
                &ldquo;{t.text}&rdquo;
              </p>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-purple-500 flex items-center justify-center text-white font-bold text-sm">
                  {t.name[0]}
                </div>
                <div>
                  <div className="font-bold text-sm">{t.name}</div>
                  <div className="text-xs text-muted-foreground">
                    {t.role} at {t.company}
                  </div>
                </div>
              </div>
            </AppleCard>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
