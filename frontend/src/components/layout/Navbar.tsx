"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, Menu, X, Link2, FileText, MessageSquare, LayoutDashboard, Mic, Target, ListChecks, ArrowRight, Send, User } from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { ThemeToggle } from "@/components/ui/ThemeToggle";

// Navigation items configuration
const navItems = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard, gradient: "from-blue-500 to-cyan-400" },
  { href: "/jobs", label: "Jobs", icon: Link2, gradient: "from-purple-500 to-pink-400" },
  { href: "/fit-analysis", label: "Fit", icon: Target, gradient: "from-orange-500 to-red-400" },
  { href: "/resumes", label: "Resumes", icon: FileText, gradient: "from-indigo-500 to-violet-400" },
  { href: "/bullet-library", label: "Bullets", icon: ListChecks, gradient: "from-green-500 to-emerald-400" },
  { href: "/interview", label: "Interview", icon: Mic, gradient: "from-pink-500 to-rose-400" },
  { href: "/outreach", label: "Outreach", icon: Send, gradient: "from-cyan-500 to-blue-400" },
  { href: "/communication", label: "Messages", icon: MessageSquare, gradient: "from-sky-500 to-indigo-400" },
];

export function Navbar() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const mountedRef = useRef(false);

  // Track mount state without re-render
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  useEffect(() => {
    if (!mobileMenuOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileMenuOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [mobileMenuOpen]);

  // One update per frame, and a passive listener so scrolling is not blocked.
  useEffect(() => {
    let frame = 0;
    const handleScroll = () => {
      if (frame) return;
      frame = window.requestAnimationFrame(() => {
        frame = 0;
        const next = window.scrollY > 20;
        setScrolled((current) => (current === next ? current : next));
      });
    };
    handleScroll();
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", handleScroll);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, []);

  // Use consistent base class for SSR to avoid hydration mismatch
  // Always use non-scrolled style on initial render, then update after hydration
  const navClass = scrolled
    ? "bg-background/80 backdrop-blur-xl border-b border-border/40 shadow-lg shadow-primary/5"
    : "bg-background/60 backdrop-blur-md border-b border-border/20";

  return (
    <motion.nav
      aria-label="Primary"
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${navClass}`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-16 sm:h-18">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2.5 group relative z-10">
            <motion.div
              whileHover={{ scale: 1.05, rotate: 5 }}
              whileTap={{ scale: 0.95 }}
              className="relative"
            >
              <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-primary via-blue-500 to-purple-600 flex items-center justify-center shadow-xl shadow-primary/30 group-hover:shadow-2xl group-hover:shadow-primary/40 transition-all duration-300">
                <Sparkles className="w-5 h-5 text-white drop-shadow-lg" />
              </div>
              {/* Glow effect */}
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-primary to-purple-600 blur-xl opacity-50 group-hover:opacity-75 transition-opacity -z-10" />
            </motion.div>
            <span className="text-lg sm:text-xl font-bold tracking-tight">
              Career<span className="bg-gradient-to-r from-primary via-blue-500 to-purple-500 bg-clip-text text-transparent">Agent</span>
              <span className="bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-transparent">Pro</span>
            </span>
          </Link>

          <div className="hidden lg:flex flex-1 justify-center items-center px-8">
            <div className="flex items-center gap-1 bg-secondary/50 backdrop-blur-sm rounded-full px-2 py-1.5 border border-border/30 shadow-sm">
              {navItems.map((item) => {
                const isActive = pathname === item.href;
                return (
                  <Link key={item.href} href={item.href}>
                    <motion.div
                      whileHover={{ scale: 1.03 }}
                      whileTap={{ scale: 0.97 }}
                      className="relative group"
                    >
                      <div
                        className={`px-3 lg:px-4 py-2 rounded-full text-[13px] font-medium transition-all relative flex items-center gap-1.5 ${isActive
                          ? "text-foreground shadow-sm"
                          : "text-muted-foreground hover:text-foreground"
                          }`}
                      >
                        <item.icon size={14} className={isActive ? "text-primary" : ""} />
                        <span>{item.label}</span>
                        {isActive && (
                          <motion.div
                            layoutId="navbar-indicator"
                            className={`absolute inset-0 bg-gradient-to-r ${item.gradient} opacity-[0.08] rounded-full -z-10 border border-primary/10`}
                            transition={{ type: "spring", stiffness: 400, damping: 30 }}
                          />
                        )}
                      </div>
                      {/* Hover gradient effect */}
                      {!isActive && (
                        <div className={`absolute inset-0 bg-gradient-to-r ${item.gradient} opacity-0 group-hover:opacity-[0.05] rounded-full transition-opacity`} />
                      )}
                    </motion.div>
                  </Link>
                );
              })}
            </div>
          </div>

          {/* Right Side - Theme Toggle & CTA */}
          <div className="flex items-center gap-2 sm:gap-3">
            {/* Theme Toggle */}
            <ThemeToggle />

            {/* Profile Link (Icon) */}
            <Link href="/profile" aria-label="Profile" className="hidden sm:flex items-center justify-center w-9 h-9 rounded-full bg-secondary/80 hover:bg-secondary text-muted-foreground hover:text-foreground transition-all">
              <User size={18} />
            </Link>

            {/* CTA Button - Desktop (Hide on Jobs page) */}
            {pathname !== "/jobs" && (
              <div className="hidden sm:block">
                <Link href="/jobs">
                  <motion.button
                    whileHover={{ scale: 1.02, boxShadow: "0 10px 30px -5px rgba(0, 113, 227, 0.35)" }}
                    whileTap={{ scale: 0.98 }}
                    className="group relative inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-gradient-to-r from-primary via-blue-600 to-purple-600 text-white font-semibold text-sm shadow-lg shadow-primary/25 overflow-hidden"
                  >
                    <span className="relative z-10">Analyze Job</span>
                    <ArrowRight size={14} className="relative z-10 group-hover:rotate-12 transition-transform" />
                    {/* Animated shine effect */}
                    <div className="absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-1000 bg-gradient-to-r from-transparent via-white/20 to-transparent" />
                  </motion.button>
                </Link>
              </div>
            )}

            {/* Mobile Menu Toggle */}
            <motion.button
              whileTap={{ scale: 0.95 }}
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="lg:hidden p-2.5 rounded-xl hover:bg-secondary/80 active:bg-secondary transition-colors"
              aria-label={mobileMenuOpen ? "Close menu" : "Open menu"}
              aria-expanded={mobileMenuOpen}
              aria-controls="mobile-nav"
            >
              <AnimatePresence mode="wait">
                <motion.span
                  key={mobileMenuOpen ? "close" : "open"}
                  initial={{ opacity: 0, rotate: -90 }}
                  animate={{ opacity: 1, rotate: 0 }}
                  exit={{ opacity: 0, rotate: 90 }}
                  transition={{ duration: 0.2 }}
                >
                  {mobileMenuOpen ? <X size={22} /> : <Menu size={22} />}
                </motion.span>
              </AnimatePresence>
            </motion.button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            id="mobile-nav"
            className="lg:hidden border-t border-border/30 bg-background/95 backdrop-blur-xl overflow-hidden"
          >
            <div className="px-4 py-5 space-y-1 max-w-lg mx-auto">
              {navItems.map((item, index) => {
                const isActive = pathname === item.href;
                return (
                  <motion.div
                    key={item.href}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                  >
                    <Link
                      href={item.href}
                      onClick={() => setMobileMenuOpen(false)}
                      className="block"
                    >
                      <div
                        className={`group relative flex items-center gap-3 px-4 py-3.5 rounded-2xl text-sm font-medium transition-all ${isActive
                          ? `bg-gradient-to-r ${item.gradient} bg-opacity-10 text-foreground shadow-sm border border-primary/10`
                          : "text-foreground hover:bg-secondary/80"
                          }`}
                      >
                        <div className={`w-9 h-9 rounded-xl bg-gradient-to-br ${item.gradient} flex items-center justify-center shadow-md ${!isActive && "opacity-60 group-hover:opacity-100"} transition-opacity`}>
                          <item.icon size={18} className="text-white" />
                        </div>
                        <span>{item.label}</span>
                        {isActive && (
                          <div className="ml-auto w-2 h-2 rounded-full bg-primary animate-pulse" />
                        )}
                      </div>
                    </Link>
                  </motion.div>
                );
              })}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="pt-4"
              >
                <Link href="/jobs" onClick={() => setMobileMenuOpen(false)}>
                  <button className="w-full relative overflow-hidden inline-flex items-center justify-center gap-2 px-5 py-4 rounded-2xl bg-gradient-to-r from-primary via-blue-600 to-purple-600 text-white font-semibold text-sm shadow-lg shadow-primary/25">
                    <ArrowRight size={16} />
                    Analyze Job
                    {/* Animated shine */}
                    <div className="absolute inset-0 -translate-x-full animate-[shimmer_2s_infinite] bg-gradient-to-r from-transparent via-white/20 to-transparent" />
                  </button>
                </Link>
              </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Add shimmer animation to globals.css */}
      <style jsx global>{`
        @keyframes shimmer {
          0% {
            transform: translateX(-100%);
          }
          100% {
            transform: translateX(100%);
          }
        }
      `}</style>
    </motion.nav>
  );
}
