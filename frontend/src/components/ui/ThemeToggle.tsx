"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sun, Moon, Monitor } from "lucide-react";

type Theme = "light" | "dark" | "system";

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");
  const [mounted, setMounted] = useState(false);

  const applyTheme = useCallback((newTheme: Theme) => {
    const root = document.documentElement;

    if (newTheme === "system") {
      const prefersDark = window.matchMedia(
        "(prefers-color-scheme: dark)",
      ).matches;
      root.classList.toggle("dark", prefersDark);
    } else {
      root.classList.toggle("dark", newTheme === "dark");
    }
  }, []);

  useEffect(() => {
    // Use requestAnimationFrame to avoid synchronous setState warning
    requestAnimationFrame(() => {
      setMounted(true);
      const stored = localStorage.getItem("theme") as Theme;
      if (stored) {
        setTheme(stored);
        applyTheme(stored);
      } else {
        applyTheme("system");
      }
    });
  }, [applyTheme]);

  const cycleTheme = () => {
    const themes: Theme[] = ["light", "dark", "system"];
    const currentIndex = themes.indexOf(theme);
    const nextTheme = themes[(currentIndex + 1) % themes.length];

    setTheme(nextTheme);
    localStorage.setItem("theme", nextTheme);
    applyTheme(nextTheme);
  };

  if (!mounted) {
    return (
      <div className="w-10 h-10 rounded-full bg-secondary animate-pulse" />
    );
  }

  const iconProps = { size: 18, strokeWidth: 2 };
  const icons = {
    light: <Sun {...iconProps} />,
    dark: <Moon {...iconProps} />,
    system: <Monitor {...iconProps} />,
  };

  return (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={cycleTheme}
      className="relative w-10 h-10 rounded-full bg-secondary flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-secondary/80 transition-colors"
      aria-label={`Switch theme (current: ${theme})`}
      title={`Theme: ${theme}`}
    >
      <AnimatePresence mode="wait">
        <motion.span
          key={theme}
          initial={{ opacity: 0, rotate: -90, scale: 0.5 }}
          animate={{ opacity: 1, rotate: 0, scale: 1 }}
          exit={{ opacity: 0, rotate: 90, scale: 0.5 }}
          transition={{ duration: 0.2 }}
        >
          {icons[theme]}
        </motion.span>
      </AnimatePresence>
    </motion.button>
  );
}
