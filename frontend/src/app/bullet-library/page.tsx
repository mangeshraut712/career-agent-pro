"use client";

import { FADE_IN } from "@/lib/animations";
import { JobData } from "@/types/shared";
import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  ListChecks,
  Plus,
  Tag,
  Search,
  Trash2,
  Edit2,
  Check,
  X,
  Sparkles,
  Target,
  Copy,
} from "lucide-react";
import axios from "axios";
import API_URL from "@/lib/api";
import { AppleCard } from "@/components/ui/AppleCard";
import { AppleButton } from "@/components/ui/AppleButton";
import { useToast } from "@/components/ui/Toast";
import { secureGet, secureSet } from "@/lib/secureStorage";
import { STORAGE_KEYS } from "@/lib/storageKeys";

interface BulletItem {
  id: string;
  text: string;
  tags: string[];
  competency?: string;
  createdAt: string;
}

interface SelectedBullet {
  id: string;
  text: string;
  competency: string;
  score: number;
  analysis_score: number;
  matched_keywords: string[];
}

const COMPETENCY_OPTIONS = [
  { value: "technical_skills", label: "Technical Skills" },
  { value: "leadership", label: "Leadership" },
  { value: "product_strategy", label: "Product Strategy" },
  { value: "communication", label: "Communication" },
  { value: "execution", label: "Execution" },
];

export default function BulletLibraryPage() {
  const { toast } = useToast();
  const [bullets, setBullets] = useState<BulletItem[]>([]);
  const [currentJob, setCurrentJob] = useState<JobData | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterCompetency, setFilterCompetency] = useState("all");
  const [newBulletText, setNewBulletText] = useState("");
  const [newBulletTags, setNewBulletTags] = useState("");
  const [newBulletCompetency, setNewBulletCompetency] =
    useState("product_strategy");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState("");
  const [editingTags, setEditingTags] = useState("");
  const [editingCompetency, setEditingCompetency] =
    useState("product_strategy");
  const [jobOverride, setJobOverride] = useState("");
  const [selectionResult, setSelectionResult] = useState<SelectedBullet[]>([]);
  const [distribution, setDistribution] = useState<Record<string, number>>({});
  const [isSelecting, setIsSelecting] = useState(false);

  useEffect(() => {
    const savedBullets = secureGet<BulletItem[]>(STORAGE_KEYS.BULLET_LIBRARY);
    if (savedBullets) {
      setBullets(savedBullets);
    }
    const savedJob = secureGet<JobData>(STORAGE_KEYS.CURRENT_JOB_FOR_RESUME);
    if (savedJob) {
      setCurrentJob(savedJob);
    }
  }, []);

  const saveBullets = (nextBullets: BulletItem[]) => {
    setBullets(nextBullets);
    secureSet(STORAGE_KEYS.BULLET_LIBRARY, nextBullets);
  };

  const filteredBullets = useMemo(() => {
    return bullets.filter((bullet) => {
      const matchesSearch =
        bullet.text.toLowerCase().includes(searchTerm.toLowerCase()) ||
        bullet.tags.join(" ").toLowerCase().includes(searchTerm.toLowerCase());
      const matchesCompetency =
        filterCompetency === "all" || bullet.competency === filterCompetency;
      return matchesSearch && matchesCompetency;
    });
  }, [bullets, searchTerm, filterCompetency]);

  const handleAddBullet = () => {
    if (!newBulletText.trim()) {
      toast("Add a bullet before saving", "error");
      return;
    }

    const tags = newBulletTags
      .split(",")
      .map((tag) => tag.trim())
      .filter(Boolean);
    const newBullet: BulletItem = {
      id: Math.random().toString(36).slice(2),
      text: newBulletText.trim(),
      tags,
      competency: newBulletCompetency,
      createdAt: new Date().toISOString(),
    };

    saveBullets([newBullet, ...bullets]);
    setNewBulletText("");
    setNewBulletTags("");
    toast("Bullet saved to library", "success");
  };

  const startEditing = (bullet: BulletItem) => {
    setEditingId(bullet.id);
    setEditingText(bullet.text);
    setEditingTags(bullet.tags.join(", "));
    setEditingCompetency(bullet.competency || "product_strategy");
  };

  const cancelEditing = () => {
    setEditingId(null);
    setEditingText("");
    setEditingTags("");
  };

  const saveEditing = (bulletId: string) => {
    const tags = editingTags
      .split(",")
      .map((tag) => tag.trim())
      .filter(Boolean);
    const updated = bullets.map((bullet) => {
      if (bullet.id !== bulletId) return bullet;
      return {
        ...bullet,
        text: editingText.trim(),
        tags,
        competency: editingCompetency,
      };
    });
    saveBullets(updated);
    cancelEditing();
    toast("Bullet updated", "success");
  };

  const deleteBullet = (bulletId: string) => {
    const updated = bullets.filter((bullet) => bullet.id !== bulletId);
    saveBullets(updated);
    toast("Bullet removed", "info");
  };

  const buildJobDescription = () => {
    if (jobOverride.trim()) {
      return jobOverride.trim();
    }
    if (!currentJob) return "";

    return [
      `${currentJob.title} at ${currentJob.company}`,
      currentJob.description,
      ...(currentJob.responsibilities || []),
      ...(currentJob.requirements || []),
      ...(currentJob.skills || []),
    ]
      .filter(Boolean)
      .join("\n");
  };

  const selectForJob = async () => {
    if (!bullets.length) {
      toast("Add bullets to your library first", "error");
      return;
    }

    const jobDescription = buildJobDescription();
    if (!jobDescription) {
      toast("Add a job description or analyze a job first", "error");
      return;
    }

    setIsSelecting(true);
    try {
      const response = await axios.post(
        `${API_URL}/bullet-library/select-for-job`,
        {
          job_description: jobDescription,
          bullets: bullets.map((bullet) => ({
            id: bullet.id,
            text: bullet.text,
            tags: bullet.tags,
            competency: bullet.competency,
          })),
          count: 13,
        },
      );

      setSelectionResult(response.data.selected_bullets || []);
      setDistribution(response.data.distribution || {});
      toast("Best-fit bullets selected", "success");
    } catch (error) {
      console.error(error);
      toast("Failed to select bullets", "error");
    } finally {
      setIsSelecting(false);
    }
  };

  const copySelected = async () => {
    const text = selectionResult.map((bullet) => `• ${bullet.text}`).join("\n");
    await navigator.clipboard.writeText(text);
    toast("Selected bullets copied", "success");
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background via-background to-secondary/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-12 lg:py-20">
        {/* Immersive Hero Header */}
        <motion.div {...FADE_IN} className="mb-12 relative group">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-2xl bg-amber-500/10 flex items-center justify-center text-amber-500 shadow-lg shadow-amber-500/5">
              <Target size={24} />
            </div>
            <h2 className="text-sm font-black uppercase tracking-[0.2em] text-amber-500/80">
              Impact Arsenal
            </h2>
          </div>
          <h1 className="text-5xl sm:text-6xl font-bold tracking-tight mb-6">
            Bullet <span className="text-amber-500">Library.</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl leading-relaxed">
            Build a repository of high-impact achievements. Auto-select the
            perfect points for every application.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-8">
            <motion.div {...FADE_IN}>
              <AppleCard className="p-8 border-border/40 bg-card/60 backdrop-blur-sm rounded-[2.5rem] shadow-xl">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-2xl bg-primary/10 flex items-center justify-center">
                    <ListChecks className="w-5 h-5 text-primary" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold">Add Bullet</h2>
                    <p className="text-sm text-muted-foreground">
                      240-260 characters works best
                    </p>
                  </div>
                </div>

                <div className="space-y-4">
                  <textarea
                    value={newBulletText}
                    onChange={(e) => setNewBulletText(e.target.value)}
                    rows={4}
                    className="w-full rounded-2xl border border-border bg-background px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30"
                    placeholder="Led cross-functional discovery for payment reconciliation platform..."
                  />
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="relative">
                      <Tag className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                      <input
                        value={newBulletTags}
                        onChange={(e) => setNewBulletTags(e.target.value)}
                        className="w-full rounded-xl border border-border bg-background pl-9 pr-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30"
                        placeholder="tags: growth, analytics, leadership"
                      />
                    </div>
                    <select
                      value={newBulletCompetency}
                      onChange={(e) => setNewBulletCompetency(e.target.value)}
                      className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30"
                    >
                      {COMPETENCY_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <AppleButton onClick={handleAddBullet} variant="primary">
                    <Plus size={16} />
                    Save Bullet
                  </AppleButton>
                </div>
              </AppleCard>
            </motion.div>

            <motion.div {...FADE_IN}>
              <AppleCard className="p-8 border-border/40 bg-card/60 backdrop-blur-sm rounded-[2.5rem] shadow-xl">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
                  <div>
                    <h2 className="text-lg font-semibold">Library</h2>
                    <p className="text-sm text-muted-foreground">
                      {bullets.length} bullets saved
                    </p>
                  </div>
                  <div className="flex gap-3 w-full md:w-auto">
                    <div className="relative flex-1 md:w-56">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                      <input
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full rounded-xl border border-border bg-background pl-9 pr-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30"
                        placeholder="Search bullets"
                      />
                    </div>
                    <select
                      value={filterCompetency}
                      onChange={(e) => setFilterCompetency(e.target.value)}
                      className="rounded-xl border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30"
                    >
                      <option value="all">All</option>
                      {COMPETENCY_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                <div className="space-y-4">
                  {filteredBullets.length === 0 && (
                    <div className="text-sm text-muted-foreground">
                      No bullets yet. Add your first bullet above.
                    </div>
                  )}
                  {filteredBullets.map((bullet) => (
                    <div
                      key={bullet.id}
                      className="rounded-2xl border border-border/70 bg-secondary/30 p-4"
                    >
                      {editingId === bullet.id ? (
                        <div className="space-y-3">
                          <textarea
                            value={editingText}
                            onChange={(e) => setEditingText(e.target.value)}
                            rows={3}
                            className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30"
                          />
                          <div className="grid md:grid-cols-2 gap-3">
                            <input
                              value={editingTags}
                              onChange={(e) => setEditingTags(e.target.value)}
                              className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30"
                              placeholder="tags: growth, analytics"
                            />
                            <select
                              value={editingCompetency}
                              onChange={(e) =>
                                setEditingCompetency(e.target.value)
                              }
                              className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30"
                            >
                              {COMPETENCY_OPTIONS.map((option) => (
                                <option key={option.value} value={option.value}>
                                  {option.label}
                                </option>
                              ))}
                            </select>
                          </div>
                          <div className="flex gap-2">
                            <AppleButton
                              size="sm"
                              onClick={() => saveEditing(bullet.id)}
                            >
                              <Check size={14} />
                              Save
                            </AppleButton>
                            <AppleButton
                              size="sm"
                              variant="secondary"
                              onClick={cancelEditing}
                            >
                              <X size={14} />
                              Cancel
                            </AppleButton>
                          </div>
                        </div>
                      ) : (
                        <div className="space-y-3">
                          <p className="text-sm leading-relaxed">
                            {bullet.text}
                          </p>
                          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                            <span className="rounded-full bg-background px-3 py-1">
                              {COMPETENCY_OPTIONS.find(
                                (c) => c.value === bullet.competency,
                              )?.label || "General"}
                            </span>
                            {bullet.tags.map((tag) => (
                              <span
                                key={tag}
                                className="rounded-full bg-background px-3 py-1"
                              >
                                #{tag}
                              </span>
                            ))}
                          </div>
                          <div className="flex gap-2">
                            <button
                              onClick={() => startEditing(bullet)}
                              className="text-xs font-medium text-primary flex items-center gap-1"
                            >
                              <Edit2 size={12} /> Edit
                            </button>
                            <button
                              onClick={() => deleteBullet(bullet.id)}
                              className="text-xs font-medium text-destructive flex items-center gap-1"
                            >
                              <Trash2 size={12} /> Delete
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </AppleCard>
            </motion.div>
          </div>

          <div className="space-y-6">
            <motion.div {...FADE_IN}>
              <AppleCard className="p-8 border-border/40 bg-card/60 backdrop-blur-sm rounded-[2.5rem] shadow-xl">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-2xl bg-primary/10 flex items-center justify-center">
                    <Target className="w-5 h-5 text-primary" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold">Auto-Select</h2>
                    <p className="text-sm text-muted-foreground">
                      Use your latest job or paste a JD
                    </p>
                  </div>
                </div>

                <textarea
                  value={jobOverride}
                  onChange={(e) => setJobOverride(e.target.value)}
                  rows={6}
                  className="w-full rounded-2xl border border-border bg-background px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 mb-4"
                  placeholder={
                    currentJob
                      ? "Optional: paste a different job description"
                      : "Paste a job description to select bullets"
                  }
                />

                <AppleButton
                  onClick={selectForJob}
                  disabled={isSelecting}
                  variant="primary"
                  className="w-full"
                >
                  <Sparkles size={16} />
                  {isSelecting ? "Selecting..." : "Select Best Bullets"}
                </AppleButton>
              </AppleCard>
            </motion.div>

            <motion.div {...FADE_IN}>
              <AppleCard className="p-8 border-border/40 bg-card/60 backdrop-blur-sm rounded-[2.5rem] shadow-xl">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold">Selected Bullets</h3>
                  {selectionResult.length > 0 && (
                    <button
                      onClick={copySelected}
                      className="text-xs font-medium text-primary flex items-center gap-1"
                    >
                      <Copy size={12} /> Copy
                    </button>
                  )}
                </div>

                {Object.keys(distribution).length > 0 && (
                  <div className="mb-4 text-xs text-muted-foreground">
                    {Object.entries(distribution).map(([key, value]) => (
                      <span key={key} className="inline-flex items-center mr-2">
                        {COMPETENCY_OPTIONS.find((c) => c.value === key)
                          ?.label || key}
                        : {value}
                      </span>
                    ))}
                  </div>
                )}

                <div className="space-y-3">
                  {selectionResult.length === 0 ? (
                    <p className="text-sm text-muted-foreground">
                      No selection yet.
                    </p>
                  ) : (
                    selectionResult.map((bullet) => (
                      <div
                        key={bullet.id}
                        className="rounded-xl border border-border/70 bg-secondary/30 p-3"
                      >
                        <p className="text-sm leading-relaxed mb-2">
                          {bullet.text}
                        </p>
                        <div className="text-xs text-muted-foreground flex flex-wrap gap-2">
                          <span>Score: {bullet.score}</span>
                          <span>Framework: {bullet.analysis_score}</span>
                          <span>
                            {COMPETENCY_OPTIONS.find(
                              (c) => c.value === bullet.competency,
                            )?.label || bullet.competency}
                          </span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </AppleCard>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
