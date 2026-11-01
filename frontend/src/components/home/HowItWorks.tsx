"use client";

import { AppleCard } from "@/components/ui/AppleCard";
import { Link2, Sparkles, Send } from "lucide-react";

const steps = [
  {
    icon: Link2,
    title: "1. Paste Job URL",
    desc: "Simply paste the link to any job posting you want to apply for. We support all major job boards.",
    color: "text-blue-500",
    bg: "bg-blue-500/10",
  },
  {
    icon: Sparkles,
    title: "2. AI Optimization",
    desc: "Our engine analyzes the job against your profile, identifying gaps and tailoring your resume keywords instantly.",
    color: "text-purple-500",
    bg: "bg-purple-500/10",
  },
  {
    icon: Send,
    title: "3. Apply with Confidence",
    desc: "Get a match score, a tailored resume, and a custom cover letter. Apply knowing you've beaten the ATS.",
    color: "text-green-500",
    bg: "bg-green-500/10",
  },
];

export function HowItWorks() {
  return (
    <section className="py-32 bg-background border-t border-border/40">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center max-w-3xl mx-auto mb-20">
          <h2 className="text-4xl font-bold tracking-tight mb-4">
            How it works
          </h2>
          <p className="text-xl text-muted-foreground">
            Three simple steps to your next interview.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 relative">
          {/* Connector Line (Desktop) */}
          <div className="hidden md:block absolute top-12 left-[16%] right-[16%] h-0.5 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-green-500/20 -z-10" />

          {steps.map((step, i) => (
            <div key={i} className="relative group">
              <AppleCard className="h-full p-8 flex flex-col items-center text-center border-border/40 hover:border-primary/20 transition-all hover:-translate-y-2">
                <div
                  className={`w-24 h-24 rounded-[2rem] ${step.bg} ${step.color} flex items-center justify-center mb-8 shadow-lg group-hover:scale-110 transition-transform`}
                >
                  <step.icon size={40} />
                </div>
                <h3 className="text-2xl font-bold mb-4">{step.title}</h3>
                <p className="text-muted-foreground leading-relaxed">
                  {step.desc}
                </p>
              </AppleCard>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
