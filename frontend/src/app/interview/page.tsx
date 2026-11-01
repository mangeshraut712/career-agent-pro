"use client";

import { useState, useEffect } from "react";
import { FADE_IN } from "@/lib/animations";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mic,
  Brain,
  MessageCircle,
  Sparkles,
  ChevronRight,
  Target,
  BookOpen,
  Lightbulb,
  Clock,
  Star,
  RefreshCw,
  AlertCircle,
  Building2,
  Headphones,
  Square,
} from "lucide-react";
import { AppleCard } from "@/components/ui/AppleCard";
import { AppleButton } from "@/components/ui/AppleButton";
import { useToast } from "@/components/ui/Toast";
import { secureGet } from "@/lib/secureStorage";
import { STORAGE_KEYS } from "@/lib/storageKeys";

interface InterviewQuestion {
  id: string;
  category: string;
  question: string;
  difficulty: "Easy" | "Medium" | "Hard";
  tips: string[];
  sampleAnswer?: string;
}

const categories = [
  {
    id: "behavioral",
    label: "Behavioral",
    icon: MessageCircle,
    color: "bg-blue-500",
    gradient: "from-blue-500 to-indigo-600",
    desc: "STAR Method focus",
  },
  {
    id: "technical",
    label: "Technical",
    icon: Brain,
    color: "bg-purple-500",
    gradient: "from-purple-500 to-pink-500",
    desc: "System Design & Logic",
  },
  {
    id: "situational",
    label: "Situational",
    icon: Target,
    color: "bg-emerald-500",
    gradient: "from-emerald-500 to-teal-600",
    desc: "Hypothetical Scenarios",
  },
  {
    id: "culture",
    label: "Culture Fit",
    icon: Star,
    color: "bg-amber-500",
    gradient: "from-amber-500 to-orange-600",
    desc: "Values & Alignment",
  },
];

const sampleQuestions: InterviewQuestion[] = [
  {
    id: "1",
    category: "behavioral",
    question:
      "Tell me about a time when you had to deal with a difficult team member.",
    difficulty: "Medium",
    tips: [
      "Use the STAR method (Situation, Task, Action, Result)",
      "Focus on how you resolved the conflict professionally",
      "Highlight what you learned from the experience",
    ],
    sampleAnswer:
      "In my previous role, I worked with a colleague who often missed deadlines. I scheduled a private conversation to understand their challenges, discovered they were overwhelmed with tasks. Together we prioritized their workload and I offered to help with some items. This improved our team's delivery time by 30%.",
  },
  {
    id: "2",
    category: "technical",
    question:
      "How would you design a scalable system to handle millions of concurrent users?",
    difficulty: "Hard",
    tips: [
      "Discuss horizontal scaling and load balancing",
      "Mention caching strategies (Redis, CDN)",
      "Talk about database sharding and replication",
      "Consider microservices architecture",
    ],
  },
  {
    id: "3",
    category: "situational",
    question: "How would you prioritize tasks when everything seems urgent?",
    difficulty: "Easy",
    tips: [
      "Mention using frameworks like Eisenhower Matrix",
      "Discuss communication with stakeholders",
      "Show how you delegate when appropriate",
    ],
  },
  {
    id: "4",
    category: "culture",
    question: "Why do you want to work at our company?",
    difficulty: "Medium",
    tips: [
      "Research the company's mission and values",
      "Connect your experience to their goals",
      "Show genuine enthusiasm and specific reasons",
    ],
  },
];

const VisualizerBar = ({ delay }: { delay: number }) => (
  <motion.div
    animate={{ height: [10, 30, 15, 40, 10] }}
    transition={{ duration: 1, repeat: Infinity, delay, ease: "easeInOut" }}
    className="w-1.5 bg-primary/40 rounded-full"
  />
);

export default function InterviewPrepPage() {
  const { toast } = useToast();
  const [selectedCategory, setSelectedCategory] =
    useState<string>("behavioral");
  const [currentQuestion, setCurrentQuestion] =
    useState<InterviewQuestion | null>(null);
  const [showAnswer, setShowAnswer] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  // Practice mode tracking
  const [currentJob, setCurrentJob] = useState<{
    title: string;
    company: string;
  } | null>(null);

  useEffect(() => {
    // Load saved job info on mount only (secureGet adds 'cap_' prefix automatically)
    const saved = secureGet<{ title: string; company: string }>(
      STORAGE_KEYS.CURRENT_JOB_FOR_RESUME,
    );
    if (saved) {
      // Use a callback to avoid lint warning about setState in effect
      requestAnimationFrame(() => {
        setCurrentJob(saved);
      });
    }
  }, []);

  // Update question when category changes
  useEffect(() => {
    const filtered = sampleQuestions.filter(
      (q) => q.category === selectedCategory,
    );
    if (filtered.length > 0) {
      requestAnimationFrame(() => {
        setCurrentQuestion(filtered[0]);
      });
    }
  }, [selectedCategory]);

  const handleCategoryChange = (catId: string) => {
    setSelectedCategory(catId);
    setShowAnswer(false);
    const filtered = sampleQuestions.filter((q) => q.category === catId);
    if (filtered.length > 0) {
      setCurrentQuestion(filtered[0]);
    }
  };

  const getNextQuestion = () => {
    setIsGenerating(true);
    setShowAnswer(false);

    setTimeout(() => {
      const filtered = sampleQuestions.filter(
        (q) => q.category === selectedCategory,
      );
      const randomIndex = Math.floor(Math.random() * filtered.length);
      setCurrentQuestion(filtered[randomIndex]);
      setIsGenerating(false);
    }, 600);
  };

  const toggleRecording = () => {
    setIsRecording(!isRecording);
    if (!isRecording) {
      toast("Listening... (Microphone Simulation)", "info");
    } else {
      toast("Analysis paused", "info");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background via-background to-secondary/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-12 lg:py-20">
        {/* Immersive Header */}
        <motion.div {...FADE_IN} className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 flex items-center justify-center text-indigo-500 shadow-lg shadow-indigo-500/5">
              <Headphones size={24} />
            </div>
            <h2 className="text-sm font-black uppercase tracking-[0.2em] text-indigo-500/80">
              Virtual Coach
            </h2>
          </div>
          <h1 className="text-5xl sm:text-6xl font-bold tracking-tight mb-6">
            Interview <span className="text-indigo-500">Simulation.</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl leading-relaxed">
            Master the art of the interview with real-time AI feedback and
            scenario-based training.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-12 gap-10">
          {/* Left Panel: Configuration & Stats */}
          <div className="lg:col-span-3 space-y-6">
            {/* Category Pills */}
            <motion.div {...FADE_IN} transition={{ delay: 0.1 }}>
              <AppleCard className="p-4 border-border/40 bg-card/60 backdrop-blur-sm rounded-[2rem]">
                <h3 className="text-xs font-black uppercase tracking-widest text-muted-foreground mb-4 pl-2">
                  Focus Area
                </h3>
                <div className="space-y-2">
                  {categories.map((cat) => {
                    const isActive = selectedCategory === cat.id;
                    return (
                      <motion.button
                        key={cat.id}
                        whileHover={{ x: 4 }}
                        onClick={() => handleCategoryChange(cat.id)}
                        className={`w-full flex items-center gap-3 p-3 rounded-xl transition-all text-left group ${
                          isActive
                            ? `bg-secondary text-foreground shadow-sm ring-1 ring-border/50`
                            : "hover:bg-secondary/50 text-muted-foreground"
                        }`}
                      >
                        <div
                          className={`w-8 h-8 rounded-lg ${isActive ? `bg-gradient-to-br ${cat.gradient} text-white` : "bg-muted text-muted-foreground"} flex items-center justify-center shadow-sm`}
                        >
                          <cat.icon size={14} />
                        </div>
                        <div>
                          <span
                            className={`block text-sm font-bold ${isActive ? "text-foreground" : ""}`}
                          >
                            {cat.label}
                          </span>
                          <span className="text-[10px] font-medium opacity-60">
                            {cat.desc}
                          </span>
                        </div>
                      </motion.button>
                    );
                  })}
                </div>
              </AppleCard>
            </motion.div>

            {/* Progress Card */}
            <motion.div {...FADE_IN} transition={{ delay: 0.2 }}>
              <AppleCard className="p-6 border-border/40 rounded-[2rem]">
                <div className="flex items-center gap-2 mb-4">
                  <Clock size={16} className="text-primary" />
                  <h3 className="text-xs font-black uppercase tracking-widest text-muted-foreground">
                    Session Stats
                  </h3>
                </div>
                <div className="space-y-4">
                  <div className="flex justify-between items-end">
                    <div className="text-3xl font-black">12</div>
                    <div className="text-xs font-bold text-emerald-500 mb-1">
                      +3 today
                    </div>
                  </div>
                  <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
                    <div className="h-full w-[60%] bg-gradient-to-r from-primary to-purple-500 rounded-full" />
                  </div>
                  <p className="text-[10px] font-medium text-muted-foreground">
                    Questions practiced this week
                  </p>
                </div>
              </AppleCard>
            </motion.div>
          </div>

          {/* Main Content: The Studio */}
          <div className="lg:col-span-9">
            <motion.div
              {...FADE_IN}
              transition={{ delay: 0.3 }}
              className="h-full"
            >
              <AppleCard className="h-full p-0 border-border/40 bg-card/80 backdrop-blur-xl rounded-[3rem] shadow-2xl overflow-hidden flex flex-col min-h-[600px] relative">
                {/* Top Bar: Context */}
                <div className="px-8 py-6 border-b border-border/20 flex items-center justify-between bg-secondary/5">
                  <div className="flex items-center gap-4">
                    {currentJob ? (
                      <div className="flex items-center gap-3 bg-secondary/50 px-4 py-2 rounded-full border border-border/50">
                        <Building2
                          size={14}
                          className="text-muted-foreground"
                        />
                        <span className="text-sm font-bold text-foreground">
                          {currentJob.company}
                        </span>
                        <span className="w-1 h-1 bg-border rounded-full" />
                        <span className="text-sm text-muted-foreground">
                          {currentJob.title}
                        </span>
                      </div>
                    ) : (
                      <div className="text-xs font-bold text-muted-foreground bg-secondary/30 px-3 py-1.5 rounded-full">
                        General Practice Mode
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`flex h-2.5 w-2.5 rounded-full ${isRecording ? "bg-rose-500 animate-pulse" : "bg-slate-300 dark:bg-slate-700"}`}
                    ></span>
                    <span className="text-xs font-bold uppercase tracking-widest text-muted-foreground">
                      {isRecording ? "Live" : "Standby"}
                    </span>
                  </div>
                </div>

                {/* Main Interaction Area */}
                <div className="flex-1 p-10 flex flex-col justify-between relative">
                  {/* Visualizer Background */}
                  {isRecording && (
                    <div className="absolute inset-0 flex items-center justify-center opacity-5 pointer-events-none">
                      <div className="flex gap-1 h-32 items-center">
                        {[...Array(20)].map((_, i) => (
                          <VisualizerBar key={i} delay={i * 0.05} />
                        ))}
                      </div>
                    </div>
                  )}

                  {currentQuestion ? (
                    <AnimatePresence mode="wait">
                      <motion.div
                        key={currentQuestion.id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="space-y-8 max-w-3xl mx-auto text-center"
                      >
                        {/* Badge */}
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-secondary/50 border border-border/50 text-[10px] font-black uppercase tracking-widest text-muted-foreground mx-auto">
                          <span
                            className={`w-2 h-2 rounded-full ${
                              currentQuestion.difficulty === "Easy"
                                ? "bg-emerald-400"
                                : currentQuestion.difficulty === "Medium"
                                  ? "bg-amber-400"
                                  : "bg-rose-400"
                            }`}
                          />
                          {currentQuestion.difficulty} Challenge
                        </div>

                        {/* Question Text */}
                        <h2 className="text-3xl md:text-4xl font-bold leading-tight tracking-tight">
                          &ldquo;{currentQuestion.question}&rdquo;
                        </h2>

                        {/* AI Coach Hints */}
                        <div className="flex flex-wrap justify-center gap-2 mt-4">
                          {currentQuestion.tips.slice(0, 2).map((tip, i) => (
                            <div
                              key={i}
                              className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-indigo-500/5 border border-indigo-500/10 text-indigo-600 font-bold text-xs"
                            >
                              <Lightbulb size={12} />
                              {tip}
                            </div>
                          ))}
                        </div>

                        {/* Sample Answer Expandable */}
                        {currentQuestion.sampleAnswer && (
                          <div className="mt-8 text-left max-w-2xl mx-auto">
                            <motion.div className="rounded-2xl border border-border/50 overflow-hidden bg-background/40 backdrop-blur-sm">
                              <button
                                onClick={() => setShowAnswer(!showAnswer)}
                                className="w-full flex items-center justify-between p-4 hover:bg-secondary/30 transition-colors"
                              >
                                <span className="flex items-center gap-2 text-sm font-bold text-muted-foreground uppercase tracking-wider">
                                  <BookOpen size={14} /> Expert Response
                                </span>
                                <ChevronRight
                                  size={16}
                                  className={`text-muted-foreground transition-transform ${showAnswer ? "rotate-90" : ""}`}
                                />
                              </button>
                              <AnimatePresence>
                                {showAnswer && (
                                  <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: "auto", opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    className="border-t border-border/30"
                                  >
                                    <div className="p-6 text-lg leading-relaxed font-medium text-foreground/80">
                                      {currentQuestion.sampleAnswer}
                                    </div>
                                  </motion.div>
                                )}
                              </AnimatePresence>
                            </motion.div>
                          </div>
                        )}
                      </motion.div>
                    </AnimatePresence>
                  ) : (
                    <div className="flex flex-col items-center justify-center h-full text-center opacity-40">
                      <AlertCircle size={48} className="mb-4" />
                      <p className="font-bold text-lg">
                        Select a category to begin
                      </p>
                    </div>
                  )}
                </div>

                {/* Bottom Control Bar */}
                <div className="p-6 border-t border-border/20 bg-secondary/5 backdrop-blur-md flex flex-wrap gap-4 items-center justify-center md:justify-between absolute bottom-0 inset-x-0">
                  <div className="hidden md:flex items-center gap-4 text-xs font-bold text-muted-foreground uppercase tracking-widest">
                    <div className="flex items-center gap-2">
                      <Sparkles size={14} className="text-indigo-500" /> AI
                      Coach Active
                    </div>
                  </div>

                  <div className="flex items-center gap-4">
                    <AppleButton
                      variant="secondary"
                      className={`h-14 w-14 rounded-full p-0 flex items-center justify-center border-border/50 ${isRecording ? "bg-rose-500/10 text-rose-500 border-rose-500/20" : ""}`}
                      onClick={toggleRecording}
                    >
                      {isRecording ? (
                        <Square size={20} fill="currentColor" />
                      ) : (
                        <Mic size={24} />
                      )}
                    </AppleButton>

                    <AppleButton
                      onClick={getNextQuestion}
                      disabled={isGenerating}
                      className="h-14 px-8 bg-foreground text-background font-bold rounded-full shadow-xl hover:scale-105 active:scale-95 transition-all text-sm uppercase tracking-widest"
                    >
                      {isGenerating ? (
                        <>
                          <RefreshCw size={16} className="animate-spin mr-2" />
                          Generating...
                        </>
                      ) : (
                        <>
                          Next scenario{" "}
                          <ChevronRight size={16} className="ml-2" />
                        </>
                      )}
                    </AppleButton>
                  </div>
                </div>
              </AppleCard>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
