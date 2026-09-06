"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    User, Mail, Phone, MapPin, Globe, Linkedin, Github, Briefcase,
    GraduationCap, Code, Save, Upload, CheckCircle2, Plus,
    Edit3, X, Heart, RefreshCw, Sparkles,
    ArrowRight, Zap
} from "lucide-react";
import API_URL from "@/lib/api";
import { AppleCard } from "@/components/ui/AppleCard";
import { AppleButton } from "@/components/ui/AppleButton";
import { useToast } from "@/components/ui/Toast";
import { secureGet, secureSet } from "@/lib/secureStorage";
import { FADE_IN } from "@/lib/animations";

interface Experience {
    id: string;
    company: string;
    role: string;
    location: string;
    startDate: string;
    endDate: string;
    type: string;
    description: string;
}

interface Education {
    id: string;
    institution: string;
    degree: string;
    field: string;
    startDate: string;
    endDate: string;
}

interface Project {
    id: string;
    name: string;
    description: string;
    technologies?: string[];
}

interface ProfileData {
    name: string;
    title: string;
    email: string;
    phone: string;
    location: string;
    linkedin: string;
    github: string;
    portfolio: string;
    jobSearchStatus: string;
    experience: Experience[];
    education: Education[];
    projects: Project[];
    certifications: string[];
    skills: string[];
    preferredSkills: string[];
    summary: string;
}

interface ParsedExperience {
    company?: string;
    role?: string;
    title?: string;
    location?: string;
    duration?: string;
    start_date?: string;
    end_date?: string;
    description?: string;
}

interface ParsedEducation {
    institution?: string;
    school?: string;
    degree?: string;
    field?: string;
    major?: string;
    graduation_year?: string;
    year?: string;
}

interface ParsedProject {
    name?: string;
    description?: string;
}

const generateId = () => Math.random().toString(36).substr(2, 9);

const EMPTY_PROFILE: ProfileData = {
    name: "",
    title: "",
    email: "",
    phone: "",
    location: "",
    linkedin: "",
    github: "",
    portfolio: "",
    jobSearchStatus: "actively_looking",
    experience: [],
    education: [],
    projects: [],
    certifications: [],
    skills: [],
    preferredSkills: [],
    summary: ""
};

const navItems = [
    { id: "info", icon: User, label: "Identity", gradient: "from-blue-500 to-cyan-400" },
    { id: "experience", icon: Briefcase, label: "Narrative", gradient: "from-purple-500 to-pink-400" },
    { id: "education", icon: GraduationCap, label: "Foundation", gradient: "from-amber-500 to-orange-400" },
    { id: "projects", icon: Zap, label: "Strategic Projects", gradient: "from-indigo-500 to-blue-400" },
    { id: "certifications", icon: Heart, label: "Professional Proof", gradient: "from-rose-500 to-orange-400" },
    { id: "skills", icon: Code, label: "Core Competency", gradient: "from-emerald-500 to-teal-400" },
];

export default function ProfilePage() {
    const { toast } = useToast();
    const [activeSection, setActiveSection] = useState("info");
    const [isSaving, setIsSaving] = useState(false);
    const [isParsing, setIsParsing] = useState(false);
    const [saveSuccess, setSaveSuccess] = useState(false);
    const [editMode, setEditMode] = useState<string | null>(null);
    const [newSkill, setNewSkill] = useState("");
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [profile, setProfile] = useState<ProfileData>(EMPTY_PROFILE);
    const [hasMounted, setHasMounted] = useState(false);
    const [parsingMessage, setParsingMessage] = useState("");

    useEffect(() => {
        setHasMounted(true);
        const saved = secureGet<ProfileData>('profile');
        if (saved) setProfile(saved);
    }, []);

    if (!hasMounted) return null;

    const handleInputChange = (field: string, value: string) => {
        setProfile(prev => ({ ...prev, [field]: value }));
    };

    const handleSave = async () => {
        setIsSaving(true);
        await new Promise(r => setTimeout(r, 600));
        secureSet('profile', profile);
        toast("Profile synchronized!", "success");
        setSaveSuccess(true);
        setTimeout(() => setSaveSuccess(false), 2000);
        setIsSaving(false);
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        setIsParsing(true);
        setParsingMessage("Initializing neural extraction...");

        try {
            const formData = new FormData();
            formData.append("file", file);

            const response = await fetch(`${API_URL}/parse-resume-stream`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            if (!response.body) throw new Error("No response body");

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let accumulatedData = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                accumulatedData += chunk;

                // Process SSE events
                const lines = accumulatedData.split("\n\n");
                accumulatedData = lines.pop() || "";

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        try {
                            const event = JSON.parse(line.substring(6));

                            if (event.status === "extracting" || event.status === "analyzing") {
                                setParsingMessage(event.message);
                            } else if (event.status === "heuristics") {
                                // Live update basic info
                                setProfile(prev => ({
                                    ...prev,
                                    email: event.data.email || prev.email,
                                    linkedin: event.data.linkedin || prev.linkedin,
                                }));
                            } else if (event.status === "completed") {
                                const parsed = event.data;
                                const mappedExperience: Experience[] = (parsed.experience || []).map((exp: ParsedExperience) => ({
                                    id: generateId(),
                                    company: exp.company || "",
                                    role: exp.role || exp.title || "",
                                    location: exp.location || "",
                                    startDate: exp.duration?.split(" - ")[0] || exp.start_date || "",
                                    endDate: exp.duration?.split(" - ")[1] || exp.end_date || "Present",
                                    type: "Full-time",
                                    description: exp.description || ""
                                }));

                                const mappedEducation: Education[] = (parsed.education || []).map((edu: ParsedEducation) => ({
                                    id: generateId(),
                                    institution: edu.institution || edu.school || "",
                                    degree: edu.degree || "",
                                    field: edu.field || edu.major || "",
                                    startDate: "",
                                    endDate: edu.graduation_year || edu.year || ""
                                }));

                                setProfile(prev => ({
                                    ...prev,
                                    name: parsed.name || prev.name,
                                    title: parsed.jobTitle || parsed.title || prev.title,
                                    email: parsed.email || prev.email,
                                    phone: parsed.phone || prev.phone,
                                    location: parsed.location || prev.location,
                                    linkedin: parsed.linkedin || prev.linkedin,
                                    github: parsed.github || prev.github,
                                    portfolio: parsed.portfolio || parsed.website || prev.portfolio,
                                    skills: parsed.skills?.length ? parsed.skills : prev.skills,
                                    summary: parsed.summary || prev.summary,
                                    experience: mappedExperience.length ? mappedExperience : prev.experience,
                                    education: mappedEducation.length ? mappedEducation : prev.education,
                                    projects: (parsed.projects || []).map((p: ParsedProject) => ({
                                        id: generateId(),
                                        name: p.name || "",
                                        description: p.description || ""
                                    })),
                                    certifications: parsed.certifications || []
                                }));
                                toast("Profile Synchronization Complete!", "success");
                            }
                        } catch (e) {
                            console.warn("Error parsing chunk", e);
                        }
                    }
                }
            }
        } catch (error: unknown) {
            console.error("Stream parse error:", error);
            toast("Extraction failed. Please try again or fill manually.", "error");
        } finally {
            setIsParsing(false);
            setParsingMessage("");
            if (fileInputRef.current) fileInputRef.current.value = "";
        }
    };

    const addExperience = () => {
        const newExp: Experience = { id: generateId(), company: "", role: "", location: "", startDate: "", endDate: "", type: "Full-time", description: "" };
        setProfile(prev => ({ ...prev, experience: [...prev.experience, newExp] }));
        setEditMode(`exp-${newExp.id}`);
    };

    const updateExperience = (id: string, field: keyof Experience, value: string) => {
        setProfile(prev => ({
            ...prev,
            experience: prev.experience.map(exp => exp.id === id ? { ...exp, [field]: value } : exp)
        }));
    };

    const deleteExperience = (id: string) => {
        setProfile(prev => ({ ...prev, experience: prev.experience.filter(exp => exp.id !== id) }));
    };

    const addEducation = () => {
        const newEdu: Education = { id: generateId(), institution: "", degree: "", field: "", startDate: "", endDate: "" };
        setProfile(prev => ({ ...prev, education: [...prev.education, newEdu] }));
        setEditMode(`edu-${newEdu.id}`);
    };

    const updateEducation = (id: string, field: keyof Education, value: string) => {
        setProfile(prev => ({
            ...prev,
            education: prev.education.map(edu => edu.id === id ? { ...edu, [field]: value } : edu)
        }));
    };

    const deleteEducation = (id: string) => {
        setProfile(prev => ({ ...prev, education: prev.education.filter(edu => edu.id !== id) }));
    };

    const addProject = () => {
        const newProj: Project = { id: generateId(), name: "", description: "" };
        setProfile(prev => ({ ...prev, projects: [...prev.projects, newProj] }));
        setEditMode(`proj-${newProj.id}`);
    };

    const updateProject = (id: string, field: keyof Project, value: string) => {
        setProfile(prev => ({
            ...prev,
            projects: prev.projects.map(p => p.id === id ? { ...p, [field]: value } : p)
        }));
    };

    const deleteProject = (id: string) => {
        setProfile(prev => ({ ...prev, projects: prev.projects.filter(p => p.id !== id) }));
    };

    const addCertification = (cert: string) => {
        if (cert.trim() && !profile.certifications.includes(cert.trim())) {
            setProfile(prev => ({ ...prev, certifications: [...prev.certifications, cert.trim()] }));
        }
    };

    const removeCertification = (cert: string) => {
        setProfile(prev => ({ ...prev, certifications: prev.certifications.filter(c => c !== cert) }));
    };

    const addSkill = () => {
        if (newSkill.trim() && !profile.skills.includes(newSkill.trim())) {
            setProfile(prev => ({ ...prev, skills: [...prev.skills, newSkill.trim()] }));
            setNewSkill("");
        }
    };

    const removeSkill = (skill: string) => {
        setProfile(prev => ({ ...prev, skills: prev.skills.filter(s => s !== skill) }));
    };

    const togglePreferred = (skill: string) => {
        setProfile(prev => ({
            ...prev,
            preferredSkills: prev.preferredSkills.includes(skill)
                ? prev.preferredSkills.filter(s => s !== skill)
                : [...prev.preferredSkills, skill]
        }));
    };

    const completeness = calculateCompleteness(profile);

    return (
        <div className="min-h-screen bg-gradient-to-b from-background via-background to-secondary/10">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 py-12 lg:py-20">

                {/* Immersive Header Card */}
                <motion.div {...FADE_IN} className="mb-12 relative group">
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10 rounded-[3rem] blur-3xl opacity-50 group-hover:opacity-75 transition-opacity -z-10" />

                    <AppleCard className="p-8 sm:p-12 border-border/40 bg-card/80 backdrop-blur-xl rounded-[3rem] shadow-2xl shadow-primary/5">
                        <div className="flex flex-col lg:flex-row items-center justify-between gap-10">
                            <div className="flex flex-col sm:flex-row items-center gap-8">
                                <div className="relative">
                                    <div className="w-32 h-32 rounded-[2.5rem] bg-gradient-to-br from-blue-500 to-cyan-600 flex items-center justify-center text-white text-5xl font-black shadow-2xl shadow-blue-500/20">
                                        {profile.name ? profile.name.split(" ").map(n => n[0]).join("").toUpperCase() : <User size={48} />}
                                    </div>
                                    <div className="absolute -bottom-2 -right-2 w-10 h-10 rounded-2xl bg-emerald-500 border-4 border-white dark:border-slate-900 flex items-center justify-center text-white shadow-lg">
                                        <CheckCircle2 size={18} />
                                    </div>
                                </div>
                                <div className="text-center sm:text-left">
                                    <h1 className="text-4xl sm:text-5xl font-black tracking-tight mb-2 leading-none">{profile.name || "Set Identity"}</h1>
                                    <p className="text-xl text-muted-foreground font-medium flex items-center justify-center sm:justify-start gap-2">
                                        <Briefcase size={18} className="text-primary" />
                                        {profile.title || "Add Primary Role"}
                                    </p>
                                </div>
                            </div>

                            <div className="flex flex-wrap justify-center gap-4">
                                <input ref={fileInputRef} type="file" accept=".pdf,.doc,.docx,.txt" onChange={handleFileUpload} className="hidden" />
                                <AppleButton
                                    variant="secondary"
                                    onClick={() => fileInputRef.current?.click()}
                                    disabled={isParsing}
                                    className="h-14 px-8 border-border/40 bg-white/50 backdrop-blur-md shadow-sm font-bold gap-4 relative min-w-[220px]"
                                >
                                    {isParsing ? (
                                        <div className="flex items-center gap-3">
                                            <RefreshCw size={18} className="animate-spin text-blue-500" />
                                            <span className="text-xs font-black uppercase tracking-widest animate-pulse">{parsingMessage || "Processing..."}</span>
                                        </div>
                                    ) : (
                                        <>
                                            <Upload size={18} />
                                            Update from Resume
                                        </>
                                    )}
                                </AppleButton>
                                <AppleButton
                                    onClick={handleSave}
                                    disabled={isSaving}
                                    className={`h-14 px-10 font-black shadow-xl shadow-primary/20 gap-2 relative overflow-hidden group ${saveSuccess ? "bg-emerald-500 shadow-emerald-500/20" : "bg-primary"}`}
                                >
                                    <div className="absolute inset-0 bg-gradient-to-r from-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                                    {isSaving ? <RefreshCw size={18} className="animate-spin" /> : saveSuccess ? <CheckCircle2 size={18} /> : <Save size={18} />}
                                    <span className="relative z-10">{isSaving ? "Syncing..." : saveSuccess ? "Verified" : "Save Profile"}</span>
                                </AppleButton>
                            </div>
                        </div>
                    </AppleCard>
                </motion.div>

                <div className="grid lg:grid-cols-12 gap-10">

                    {/* Navigation Sidebar */}
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.5, delay: 0.2 }}
                        className="lg:col-span-3 space-y-8"
                    >
                        <AppleCard className="p-3 border-border/40 rounded-[2rem] overflow-hidden bg-card/60 backdrop-blur-sm">
                            <div className="p-4 mb-2">
                                <span className="text-[10px] font-black uppercase tracking-[0.2em] text-muted-foreground/60">Career Architecture</span>
                            </div>
                            {navItems.map((item) => (
                                <motion.button
                                    key={item.id}
                                    whileHover={{ x: 6, scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                    onClick={() => setActiveSection(item.id)}
                                    className={`w-full flex items-center gap-4 p-5 rounded-2xl transition-all mb-1 ${activeSection === item.id
                                        ? `bg-gradient-to-r ${item.gradient} text-white shadow-lg shadow-primary/10`
                                        : "hover:bg-secondary/80 text-muted-foreground hover:text-foreground"
                                        }`}
                                >
                                    <item.icon size={20} className={activeSection === item.id ? "text-white" : "text-primary/60"} />
                                    <span className="font-black text-sm">{item.label}</span>
                                    {activeSection === item.id && <ArrowRight size={14} className="ml-auto" />}
                                </motion.button>
                            ))}
                        </AppleCard>

                        <AppleCard className="p-8 border-border/40 rounded-[2rem] bg-gradient-to-br from-primary/5 to-blue-500/5">
                            <div className="flex items-center justify-between mb-6">
                                <h3 className="font-black text-xs uppercase tracking-widest text-muted-foreground/60">Completeness</h3>
                                <span className="text-2xl font-black text-primary">{completeness}%</span>
                            </div>
                            <div className="h-4 bg-secondary/50 rounded-full overflow-hidden border border-border/20 relative shadow-inner">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${completeness}%` }}
                                    transition={{ duration: 1.2, ease: "circOut" }}
                                    className="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-600 rounded-full relative"
                                >
                                    <div className="absolute inset-0 bg-white/20 animate-[shimmer_2s_infinite]" />
                                </motion.div>
                            </div>
                            <p className="text-[10px] text-center mt-4 font-bold text-muted-foreground/60 leading-relaxed uppercase tracking-widest">
                                {completeness < 100 ? "Finish all nodes for AI priority" : "System fully synchronized"}
                            </p>
                        </AppleCard>
                    </motion.div>

                    {/* Main Content Area */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.98 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.5, delay: 0.3 }}
                        className="lg:col-span-9"
                    >
                        <AnimatePresence mode="wait">
                            <motion.div
                                key={activeSection}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                transition={{ duration: 0.3 }}
                            >
                                <AppleCard className="p-10 border-border/40 rounded-[3rem] shadow-xl min-h-[600px]">

                                    {/* Section 1: Identity */}
                                    {activeSection === "info" && (
                                        <div className="space-y-12">
                                            <div className="flex items-center gap-3 mb-4">
                                                <div className="w-10 h-10 rounded-xl bg-blue-500/10 flex items-center justify-center text-blue-500">
                                                    <User size={20} />
                                                </div>
                                                <h2 className="text-2xl font-black tracking-tight">Personal Identity</h2>
                                            </div>

                                            <div className="grid md:grid-cols-2 gap-x-10 gap-y-8">
                                                <InputField id="name" name="name" icon={<User size={18} />} label="Full Name" value={profile.name} onChange={v => handleInputChange("name", v)} placeholder="John Doe" autoComplete="name" />
                                                <InputField id="title" name="title" icon={<Zap size={18} />} label="Professional Headline" value={profile.title} onChange={v => handleInputChange("title", v)} placeholder="Senior Product Architect" />
                                                <InputField id="email" name="email" icon={<Mail size={18} />} label="Digital Contact" value={profile.email} onChange={v => handleInputChange("email", v)} placeholder="hello@company.com" autoComplete="email" />
                                                <InputField id="phone" name="phone" icon={<Phone size={18} />} label="Direct Line" value={profile.phone} onChange={v => handleInputChange("phone", v)} placeholder="+1 (555) 000-0000" />
                                                <InputField id="location" name="location" icon={<MapPin size={18} />} label="Primary Location" value={profile.location} onChange={v => handleInputChange("location", v)} placeholder="New York, NY" />
                                                <InputField id="linkedin" name="linkedin" icon={<Linkedin size={18} />} label="LinkedIn Node" value={profile.linkedin} onChange={v => handleInputChange("linkedin", v)} placeholder="linkedin.com/in/username" />
                                                <InputField id="github" name="github" icon={<Github size={18} />} label="GitHub Pulse" value={profile.github} onChange={v => handleInputChange("github", v)} placeholder="github.com/username" />
                                                <InputField id="portfolio" name="portfolio" icon={<Globe size={18} />} label="Work Showcase" value={profile.portfolio} onChange={v => handleInputChange("portfolio", v)} placeholder="yourportfolio.io" />
                                            </div>

                                            <div className="pt-6 border-t border-border/20">
                                                <label className="text-xs font-black uppercase tracking-[0.2em] text-muted-foreground mb-4 block">Strategic Summary</label>
                                                <textarea
                                                    value={profile.summary}
                                                    onChange={(e) => handleInputChange("summary", e.target.value)}
                                                    rows={5}
                                                    className="w-full bg-secondary/30 rounded-3xl p-6 text-base font-medium focus:outline-none focus:ring-2 focus:ring-primary/20 border border-border/30 resize-none transition-all placeholder:text-muted-foreground/30"
                                                    placeholder="Define your unique value proposition..."
                                                />
                                            </div>
                                        </div>
                                    )}

                                    {/* Section 2: Experience / Narrative */}
                                    {activeSection === "experience" && (
                                        <div className="space-y-8">
                                            <div className="flex items-center justify-between gap-4 mb-2">
                                                <div className="flex items-center gap-3">
                                                    <div className="w-10 h-10 rounded-xl bg-purple-500/10 flex items-center justify-center text-purple-500">
                                                        <Briefcase size={20} />
                                                    </div>
                                                    <h2 className="text-2xl font-black tracking-tight">Professional Narrative</h2>
                                                </div>
                                                <AppleButton onClick={addExperience} variant="secondary" className="px-5 font-bold border-purple-500/20 text-purple-600 hover:bg-purple-500 hover:text-white transition-all">
                                                    <Plus size={18} /> Add Role
                                                </AppleButton>
                                            </div>

                                            <div className="space-y-6">
                                                {profile.experience.length === 0 ? (
                                                    <EmptySection message="No narrative nodes detected." icon={History} onAdd={addExperience} color="purple" />
                                                ) : (
                                                    profile.experience.map(exp => (
                                                        <ExperienceNode
                                                            key={exp.id}
                                                            exp={exp}
                                                            isEditing={editMode === `exp-${exp.id}`}
                                                            onEdit={() => setEditMode(`exp-${exp.id}`)}
                                                            onSave={() => setEditMode(null)}
                                                            onUpdate={updateExperience}
                                                            onDelete={() => deleteExperience(exp.id)}
                                                        />
                                                    ))
                                                )}
                                            </div>
                                        </div>
                                    )}

                                    {/* Section 3: Education */}
                                    {activeSection === "education" && (
                                        <div className="space-y-8">
                                            <div className="flex items-center justify-between gap-4 mb-2">
                                                <div className="flex items-center gap-3">
                                                    <div className="w-10 h-10 rounded-xl bg-orange-500/10 flex items-center justify-center text-orange-500">
                                                        <GraduationCap size={20} />
                                                    </div>
                                                    <h2 className="text-2xl font-black tracking-tight">Academic Foundation</h2>
                                                </div>
                                                <AppleButton onClick={addEducation} variant="secondary" className="px-5 font-bold border-orange-500/20 text-orange-600 hover:bg-orange-500 hover:text-white transition-all">
                                                    <Plus size={18} /> Add Degree
                                                </AppleButton>
                                            </div>

                                            <div className="space-y-6">
                                                {profile.education.length === 0 ? (
                                                    <EmptySection message="Education baseline unidentified." icon={GraduationCap} onAdd={addEducation} color="orange" />
                                                ) : (
                                                    profile.education.map(edu => (
                                                        <EducationNode
                                                            key={edu.id}
                                                            edu={edu}
                                                            isEditing={editMode === `edu-${edu.id}`}
                                                            onEdit={() => setEditMode(`edu-${edu.id}`)}
                                                            onSave={() => setEditMode(null)}
                                                            onUpdate={updateEducation}
                                                            onDelete={() => deleteEducation(edu.id)}
                                                        />
                                                    ))
                                                )}
                                            </div>
                                        </div>
                                    )}

                                    {/* Section 4: Projects */}
                                    {activeSection === "projects" && (
                                        <div className="space-y-8">
                                            <div className="flex items-center justify-between gap-4 mb-2">
                                                <div className="flex items-center gap-3">
                                                    <div className="w-10 h-10 rounded-xl bg-indigo-500/10 flex items-center justify-center text-indigo-500">
                                                        <Zap size={20} />
                                                    </div>
                                                    <h2 className="text-2xl font-black tracking-tight">Strategic Projects</h2>
                                                </div>
                                                <AppleButton onClick={addProject} variant="secondary" className="px-5 font-bold border-indigo-500/20 text-indigo-600 hover:bg-indigo-500 hover:text-white transition-all">
                                                    <Plus size={18} /> Add Project
                                                </AppleButton>
                                            </div>

                                            <div className="space-y-6">
                                                {profile.projects.length === 0 ? (
                                                    <EmptySection message="No project nodes deployed." icon={Zap} onAdd={addProject} color="purple" />
                                                ) : (
                                                    profile.projects.map(proj => (
                                                        <ProjectNode
                                                            key={proj.id}
                                                            proj={proj}
                                                            isEditing={editMode === `proj-${proj.id}`}
                                                            onEdit={() => setEditMode(`proj-${proj.id}`)}
                                                            onSave={() => setEditMode(null)}
                                                            onUpdate={updateProject}
                                                            onDelete={() => deleteProject(proj.id)}
                                                        />
                                                    ))
                                                )}
                                            </div>
                                        </div>
                                    )}

                                    {/* Section 5: Certifications */}
                                    {activeSection === "certifications" && (
                                        <div className="space-y-10">
                                            <div className="flex items-center gap-3 mb-2">
                                                <div className="w-10 h-10 rounded-xl bg-rose-500/10 flex items-center justify-center text-rose-500">
                                                    <Heart size={20} />
                                                </div>
                                                <h2 className="text-2xl font-black tracking-tight">Professional Proof</h2>
                                            </div>

                                            <div className="p-8 rounded-[2rem] bg-secondary/20 border border-border/20">
                                                <div className="flex flex-wrap gap-3 mb-10 min-h-[100px] items-start content-start">
                                                    {profile.certifications.map((cert) => (
                                                        <motion.div
                                                            key={cert}
                                                            initial={{ scale: 0.8, opacity: 0 }}
                                                            animate={{ scale: 1, opacity: 1 }}
                                                            whileHover={{ y: -2, scale: 1.02 }}
                                                            className="flex items-center gap-2.5 px-4 py-3 rounded-2xl bg-gradient-to-br from-rose-50 to-orange-50 dark:from-rose-950/30 dark:to-orange-950/10 border border-rose-100 dark:border-rose-900/30 shadow-sm"
                                                        >
                                                            <div className="w-6 h-6 rounded-full bg-white dark:bg-rose-900/50 flex items-center justify-center shadow-sm">
                                                                <Heart size={12} className="text-rose-500 fill-rose-500/20" />
                                                            </div>
                                                            <span className="text-sm font-bold tracking-tight text-rose-900 dark:text-rose-100">{cert}</span>
                                                            <button
                                                                onClick={() => removeCertification(cert)}
                                                                className="ml-2 p-1 rounded-full hover:bg-black/10 text-muted-foreground hover:text-rose-500 transition-colors"
                                                            >
                                                                <X size={12} />
                                                            </button>
                                                        </motion.div>
                                                    ))}
                                                    {profile.certifications.length === 0 && (
                                                        <p className="text-sm font-medium text-muted-foreground/40 italic">Add certifications to verify expertise...</p>
                                                    )}
                                                </div>

                                                <div className="flex gap-4">
                                                    <div className="relative flex-1">
                                                        <Plus className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground/60" size={18} />
                                                        <input
                                                            type="text"
                                                            placeholder="Add certification (e.g. AWS Solutions Architect)..."
                                                            onKeyDown={(e) => {
                                                                if (e.key === "Enter") {
                                                                    addCertification(e.currentTarget.value);
                                                                    e.currentTarget.value = "";
                                                                }
                                                            }}
                                                            className="w-full pl-12 pr-6 py-4 bg-white dark:bg-slate-800 rounded-2xl text-base font-bold outline-none focus:ring-2 focus:ring-rose-500/20 border border-border/40 transition-all"
                                                        />
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {/* Section 6: Skills */}
                                    {activeSection === "skills" && (
                                        <div className="space-y-10">
                                            <div className="flex items-center gap-3 mb-2">
                                                <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center text-emerald-500">
                                                    <Code size={20} />
                                                </div>
                                                <h2 className="text-2xl font-black tracking-tight">Core Competencies</h2>
                                            </div>

                                            <div className="p-8 rounded-[2rem] bg-secondary/20 border border-border/20">
                                                <div className="flex items-center gap-2 mb-6 opacity-60">
                                                    <Sparkles size={14} className="text-emerald-500" />
                                                    <span className="text-[10px] font-black uppercase tracking-widest">Mark high-priority skills with Hearts</span>
                                                </div>

                                                <div className="flex flex-wrap gap-3 mb-10 min-h-[100px] items-start content-start">
                                                    {/* eslint-disable-next-line @typescript-eslint/no-unused-vars */}
                                                    {profile.skills.map((skill, _idx) => (
                                                        <motion.div
                                                            key={skill}
                                                            initial={{ scale: 0.8, opacity: 0 }}
                                                            animate={{ scale: 1, opacity: 1 }}
                                                            whileHover={{ y: -4, scale: 1.05 }}
                                                            className={`group flex items-center gap-2 px-5 py-2.5 rounded-2xl cursor-pointer transition-all border shadow-sm ${profile.preferredSkills.includes(skill)
                                                                ? "bg-gradient-to-br from-emerald-500 to-teal-600 text-white border-transparent"
                                                                : "bg-white dark:bg-slate-800 border-border/40 hover:border-emerald-500/50"
                                                                }`}
                                                            onClick={() => togglePreferred(skill)}
                                                        >
                                                            <Heart size={14} className={`transition-colors ${profile.preferredSkills.includes(skill) ? "fill-white text-white" : "text-emerald-500 group-hover:scale-110"}`} />
                                                            <span className="text-sm font-bold tracking-tight">{skill}</span>
                                                            <button
                                                                onClick={(e) => { e.stopPropagation(); removeSkill(skill); }}
                                                                className={`ml-2 p-1 rounded-full hover:bg-black/10 transition-colors ${profile.preferredSkills.includes(skill) ? "text-white/50 hover:text-white" : "text-muted-foreground hover:text-rose-500"}`}
                                                            >
                                                                <X size={12} />
                                                            </button>
                                                        </motion.div>
                                                    ))}
                                                    {profile.skills.length === 0 && (
                                                        <p className="text-sm font-medium text-muted-foreground/40 italic">Add skills to build your graph...</p>
                                                    )}
                                                </div>

                                                <div className="flex gap-4">
                                                    <div className="relative flex-1">
                                                        <Plus className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground/60" size={18} />
                                                        <input
                                                            type="text"
                                                            value={newSkill}
                                                            onChange={(e) => setNewSkill(e.target.value)}
                                                            onKeyPress={(e) => e.key === "Enter" && addSkill()}
                                                            placeholder="Add new competency..."
                                                            className="w-full pl-12 pr-6 py-4 bg-white dark:bg-slate-800 rounded-2xl text-base font-bold outline-none focus:ring-2 focus:ring-emerald-500/20 border border-border/40 transition-all"
                                                        />
                                                    </div>
                                                    <AppleButton onClick={addSkill} className="px-8 bg-emerald-500 hover:bg-emerald-600 shadow-lg shadow-emerald-500/20 font-black tracking-widest uppercase text-xs">
                                                        Deploy
                                                    </AppleButton>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </AppleCard>
                            </motion.div>
                        </AnimatePresence>
                    </motion.div>
                </div>
            </div>
        </div>
    );
}

function calculateCompleteness(profile: ProfileData): number {
    let score = 0;
    // Base Identity (20%)
    if (profile.name && profile.email) score += 20;

    // Core Pillars (60%)
    if (profile.experience.length > 0) score += 15;
    if (profile.education.length > 0) score += 15;
    if (profile.projects.length > 0) score += 15;
    if (profile.certifications.length > 0) score += 15;

    // Competency (20%)
    if (profile.skills.length >= 8) score += 20;
    else if (profile.skills.length > 0) score += 10;

    return Math.min(score, 100);
}

interface InputFieldProps {
    id: string;
    name: string;
    icon: React.ReactNode;
    label: string;
    value: string;
    onChange: (value: string) => void;
    placeholder?: string;
    autoComplete?: string;
}

function InputField({ id, name, icon, label, value, onChange, placeholder, autoComplete }: InputFieldProps) {
    return (
        <div className="group">
            <label htmlFor={id} className="text-[10px] font-black uppercase tracking-[0.2em] text-muted-foreground mb-3 block group-hover:text-primary transition-colors cursor-pointer">{label}</label>
            <div className="relative">
                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-primary/40 group-focus-within:text-primary transition-colors pointer-events-none">
                    {icon}
                </div>
                <input
                    id={id}
                    name={name}
                    type="text"
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    placeholder={placeholder}
                    autoComplete={autoComplete}
                    className="w-full pl-12 pr-5 py-4 bg-secondary/40 rounded-2xl text-base font-bold outline-none focus:ring-2 focus:ring-primary/10 border border-border/10 focus:border-primary/20 transition-all placeholder:text-muted-foreground/20"
                />
            </div>
        </div>
    );
}

interface EmptySectionProps {
    message: string;
    icon: React.ElementType;
    onAdd: () => void;
    color: string;
}

function EmptySection({ message, icon: Icon, onAdd, color }: EmptySectionProps) {
    const colors: Record<string, string> = {
        purple: "text-purple-500 bg-purple-500/5 border-purple-500/10",
        orange: "text-orange-500 bg-orange-500/5 border-orange-500/10",
    };
    return (
        <div className={`py-16 text-center border-2 border-dashed rounded-[2.5rem] ${colors[color] || "border-border"}`}>
            <Icon size={40} className="mx-auto mb-4 opacity-40" />
            <p className="font-bold text-muted-foreground mb-6">{message}</p>
            <AppleButton onClick={onAdd} variant="secondary" className="px-6 font-black tracking-widest uppercase text-xs">Execute Setup</AppleButton>
        </div>
    );
}

interface ExperienceNodeProps {
    exp: Experience;
    isEditing: boolean;
    onEdit: () => void;
    onSave: () => void;
    onUpdate: (id: string, field: keyof Experience, value: string) => void;
    onDelete: () => void;
}

function ExperienceNode({ exp, isEditing, onEdit, onSave, onUpdate, onDelete }: ExperienceNodeProps) {
    if (isEditing) {
        return (
            <div className="p-8 border-2 border-purple-500/30 rounded-[2rem] space-y-6 bg-purple-50/10 backdrop-blur-sm">
                <div className="grid md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                        <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Context</label>
                        <input value={exp.role} onChange={(e) => onUpdate(exp.id, "role", e.target.value)} placeholder="Role Title" className="w-full px-5 py-3 bg-white border border-border rounded-xl font-bold outline-none focus:ring-2 focus:ring-purple-500/20" />
                        <input value={exp.company} onChange={(e) => onUpdate(exp.id, "company", e.target.value)} placeholder="Organization" className="w-full px-5 py-3 bg-white border border-border rounded-xl font-bold outline-none focus:ring-2 focus:ring-purple-500/20" />
                    </div>
                    <div className="space-y-4">
                        <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Logistics</label>
                        <input value={exp.location} onChange={(e) => onUpdate(exp.id, "location", e.target.value)} placeholder="Location" className="w-full px-5 py-3 bg-white border border-border rounded-xl font-bold outline-none focus:ring-2 focus:ring-purple-500/20" />
                        <div className="flex gap-2">
                            <input value={exp.startDate} onChange={(e) => onUpdate(exp.id, "startDate", e.target.value)} placeholder="Start Date" className="flex-1 px-5 py-3 bg-white border border-border rounded-xl font-bold outline-none focus:ring-2 focus:ring-purple-500/20" />
                            <input value={exp.endDate} onChange={(e) => onUpdate(exp.id, "endDate", e.target.value)} placeholder="End Date" className="flex-1 px-5 py-3 bg-white border border-border rounded-xl font-bold outline-none focus:ring-2 focus:ring-purple-500/20" />
                        </div>
                    </div>
                </div>
                <div className="space-y-4">
                    <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Impact Narrative</label>
                    <textarea value={exp.description} onChange={(e) => onUpdate(exp.id, "description", e.target.value)} placeholder="Describe your results..." rows={4} className="w-full px-5 py-4 bg-white border border-border rounded-2xl font-medium outline-none focus:ring-2 focus:ring-purple-500/20 resize-none" />
                </div>
                <div className="flex justify-end items-center gap-6">
                    <button onClick={onDelete} className="text-xs font-black uppercase tracking-widest text-rose-500 hover:text-rose-600 transition-colors">Delete Node</button>
                    <AppleButton onClick={onSave} className="bg-purple-600 hover:bg-purple-700 px-8 font-black tracking-widest uppercase text-xs shadow-lg shadow-purple-500/20">Finalize</AppleButton>
                </div>
            </div>
        );
    }
    return (
        <motion.div
            whileHover={{ y: -4, scale: 1.01 }}
            onClick={onEdit}
            className="p-6 border border-border/40 rounded-[2rem] hover:border-purple-500/40 hover:bg-purple-500/[0.02] cursor-pointer transition-all flex items-start gap-6 group bg-card shadow-sm"
        >
            <div className="w-16 h-16 rounded-[1.5rem] bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center font-black text-2xl text-white shadow-lg shadow-purple-500/20 group-hover:scale-110 transition-transform">{exp.company?.charAt(0) || "?"}</div>
            <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                    <h4 className="text-lg font-black tracking-tight group-hover:text-purple-600 transition-colors">{exp.role || "Role Unlabeled"}</h4>
                    <Edit3 size={16} className="text-muted-foreground/30 opacity-0 group-hover:opacity-100 transition-all" />
                </div>
                <p className="text-sm font-bold text-muted-foreground flex items-center gap-2">
                    <Building2 size={14} className="text-purple-500/60" /> {exp.company || "Stealth Startup"}
                </p>
                <div className="flex items-center gap-4 mt-3">
                    <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground/60 bg-secondary/80 px-2.5 py-1 rounded-lg">{exp.startDate || "Date X"} — {exp.endDate || "Date Y"}</span>
                    <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground/60 bg-secondary/80 px-2.5 py-1 rounded-lg flex items-center gap-1.5"><MapPin size={10} /> {exp.location || "Remote"}</span>
                </div>
            </div>
        </motion.div>
    );
}

interface EducationNodeProps {
    edu: Education;
    isEditing: boolean;
    onEdit: () => void;
    onSave: () => void;
    onUpdate: (id: string, field: keyof Education, value: string) => void;
    onDelete: () => void;
}

interface ProjectNodeProps {
    proj: Project;
    isEditing: boolean;
    onEdit: () => void;
    onSave: () => void;
    onUpdate: (id: string, field: keyof Project, value: string) => void;
    onDelete: () => void;
}

function ProjectNode({ proj, isEditing, onEdit, onSave, onUpdate, onDelete }: ProjectNodeProps) {
    if (isEditing) {
        return (
            <div className="p-8 border-2 border-indigo-500/30 rounded-[2rem] space-y-6 bg-indigo-50/10 backdrop-blur-sm">
                <div className="space-y-4">
                    <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Project Name</label>
                    <input value={proj.name} onChange={(e) => onUpdate(proj.id, "name", e.target.value)} placeholder="e.g. AI Career Assistant" className="w-full px-5 py-3 bg-white border border-border rounded-xl font-bold outline-none focus:ring-2 focus:ring-indigo-500/20" />
                </div>
                <div className="space-y-4">
                    <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Strategic Impact</label>
                    <textarea value={proj.description} onChange={(e) => onUpdate(proj.id, "description", e.target.value)} placeholder="Describe technologies used and outcome..." rows={4} className="w-full px-5 py-4 bg-white border border-border rounded-2xl font-medium outline-none focus:ring-2 focus:ring-indigo-500/20 resize-none" />
                </div>
                <div className="flex justify-end items-center gap-6">
                    <button onClick={onDelete} className="text-xs font-black uppercase tracking-widest text-rose-500 hover:text-rose-600 transition-colors">Delete Project</button>
                    <AppleButton onClick={onSave} className="bg-indigo-600 hover:bg-indigo-700 px-8 font-black tracking-widest uppercase text-xs shadow-lg shadow-indigo-500/20">Finalize</AppleButton>
                </div>
            </div>
        );
    }
    return (
        <motion.div
            whileHover={{ y: -4, scale: 1.01 }}
            onClick={onEdit}
            className="p-6 border border-border/40 rounded-[2rem] hover:border-indigo-500/40 hover:bg-indigo-500/[0.02] cursor-pointer transition-all flex items-start gap-6 group bg-card shadow-sm"
        >
            <div className="w-16 h-16 rounded-[1.5rem] bg-gradient-to-br from-indigo-500 to-blue-600 flex items-center justify-center text-white shadow-lg shadow-indigo-500/20 group-hover:scale-110 transition-transform"><Zap size={28} /></div>
            <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                    <h4 className="text-lg font-black tracking-tight group-hover:text-indigo-600 transition-colors">{proj.name || "Unnamed Project"}</h4>
                    <Edit3 size={16} className="text-muted-foreground/30 opacity-0 group-hover:opacity-100 transition-all" />
                </div>
                <p className="text-sm font-medium text-muted-foreground line-clamp-2">{proj.description || "No description provided."}</p>
            </div>
        </motion.div>
    );
}

function EducationNode({ edu, isEditing, onEdit, onSave, onUpdate, onDelete }: EducationNodeProps) {
    if (isEditing) {
        return (
            <div className="p-8 border-2 border-orange-500/30 rounded-[2rem] space-y-6 bg-orange-50/10 backdrop-blur-sm">
                <div className="grid md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                        <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Institution</label>
                        <input value={edu.institution} onChange={(e) => onUpdate(edu.id, "institution", e.target.value)} placeholder="University Name" className="w-full px-5 py-3 bg-white border border-border rounded-xl font-bold outline-none focus:ring-2 focus:ring-orange-500/20" />
                        <input value={edu.degree} onChange={(e) => onUpdate(edu.id, "degree", e.target.value)} placeholder="Degree Type" className="w-full px-5 py-3 bg-white border border-border rounded-xl font-bold outline-none focus:ring-2 focus:ring-orange-500/20" />
                    </div>
                    <div className="space-y-4">
                        <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Academics</label>
                        <input value={edu.field} onChange={(e) => onUpdate(edu.id, "field", e.target.value)} placeholder="Field of Study" className="w-full px-5 py-3 bg-white border border-border rounded-xl font-bold outline-none focus:ring-2 focus:ring-orange-500/20" />
                        <div className="flex gap-2">
                            <input value={edu.startDate} onChange={(e) => onUpdate(edu.id, "startDate", e.target.value)} placeholder="Start Year" className="flex-1 px-5 py-3 bg-white border border-border rounded-xl font-bold outline-none focus:ring-2 focus:ring-orange-500/20" />
                            <input value={edu.endDate} onChange={(e) => onUpdate(edu.id, "endDate", e.target.value)} placeholder="Graduation" className="flex-1 px-5 py-3 bg-white border border-border rounded-xl font-bold outline-none focus:ring-2 focus:ring-orange-500/20" />
                        </div>
                    </div>
                </div>
                <div className="flex justify-end items-center gap-6 pt-4">
                    <button onClick={onDelete} className="text-xs font-black uppercase tracking-widest text-rose-500 hover:text-rose-600 transition-colors">Delete Degree</button>
                    <AppleButton onClick={onSave} className="bg-orange-600 hover:bg-orange-700 px-8 font-black tracking-widest uppercase text-xs shadow-lg shadow-orange-500/20">Finalize</AppleButton>
                </div>
            </div>
        );
    }
    return (
        <motion.div
            whileHover={{ y: -4, scale: 1.01 }}
            onClick={onEdit}
            className="p-6 border border-border/40 rounded-[2rem] hover:border-orange-500/40 hover:bg-orange-500/[0.02] cursor-pointer transition-all flex items-start gap-6 group bg-card shadow-sm"
        >
            <div className="w-16 h-16 rounded-[1.5rem] bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center text-white shadow-lg shadow-orange-500/20 group-hover:scale-110 transition-transform"><GraduationCap size={28} /></div>
            <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                    <h4 className="text-lg font-black tracking-tight group-hover:text-orange-600 transition-colors">{edu.degree || "Degree Unlisted"}</h4>
                    <Edit3 size={16} className="text-muted-foreground/30 opacity-0 group-hover:opacity-100 transition-all" />
                </div>
                <p className="text-sm font-bold text-muted-foreground mb-3">{edu.field || "General Studies"}</p>
                <p className="text-xs font-black tracking-[0.05em] text-muted-foreground/50 flex items-center gap-2 group-hover:text-orange-600/60 transition-colors uppercase tracking-[0.1em]">
                    {edu.institution || "Unknown University"}
                </p>
                <div className="mt-3">
                    <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground/60 bg-secondary/80 px-2.5 py-1 rounded-lg">{edu.startDate || "Date X"} — {edu.endDate || "Date Y"}</span>
                </div>
            </div>
        </motion.div>
    );
}

const Building2 = ({ size = 24 }: React.SVGProps<SVGSVGElement> & { size?: number }) => (
    <svg
        width={size}
        height={size}
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
    >
        <path d="M6 22V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v18" />
        <path d="M6 12H4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2" />
        <path d="M18 9h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-2" />
        <path d="M10 6h4" />
        <path d="M10 10h4" />
        <path d="M10 14h4" />
        <path d="M10 18h4" />
    </svg>
);

const History = ({ size = 24 }: React.SVGProps<SVGSVGElement> & { size?: number }) => (
    <svg
        width={size}
        height={size}
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
    >
        <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
        <path d="M3 3v5h5" />
        <path d="M12 7v5l4 2" />
    </svg>
);
