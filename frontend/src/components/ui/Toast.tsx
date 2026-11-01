"use client";

import React, { createContext, useContext, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, CheckCircle2, AlertCircle, Info } from "lucide-react";
import { cn } from "@/lib/utils";

type ToastType = "success" | "error" | "info";

interface Toast {
  id: string;
  message: string;
  type: ToastType;
}

interface ToastContextType {
  toast: (message: string, type?: ToastType) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const toast = useCallback(
    (message: string, type: ToastType = "info") => {
      const id = Math.random().toString(36).slice(2) + Date.now().toString(36);
      setToasts((prev) => [...prev, { id, message, type }]);

      // Auto dismiss
      setTimeout(() => {
        removeToast(id);
      }, 3000);
    },
    [removeToast],
  );

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-3 pointer-events-none">
        <AnimatePresence>
          {toasts.map((t) => (
            <motion.div
              key={t.id}
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95, transition: { duration: 0.2 } }}
              className={cn(
                "pointer-events-auto flex items-center gap-3 px-4 py-3 rounded-2xl shadow-lg border backdrop-blur-md min-w-[300px]",
                "bg-background/80", // Glass effect
                t.type === "success" &&
                  "border-green-500/20 text-green-600 dark:text-green-400",
                t.type === "error" &&
                  "border-red-500/20 text-red-600 dark:text-red-400",
                t.type === "info" &&
                  "border-blue-500/20 text-blue-600 dark:text-blue-400",
              )}
            >
              {t.type === "success" && <CheckCircle2 size={18} />}
              {t.type === "error" && <AlertCircle size={18} />}
              {t.type === "info" && <Info size={18} />}
              <span className="text-sm font-medium text-foreground">
                {t.message}
              </span>
              <button
                onClick={() => removeToast(t.id)}
                className="ml-auto text-muted-foreground hover:text-foreground transition-colors"
              >
                <X size={14} />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const context = useContext(ToastContext);
  if (context === undefined) {
    throw new Error("useToast must be used within a ToastProvider");
  }
  return context;
}
