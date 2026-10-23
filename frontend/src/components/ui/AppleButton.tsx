import React from "react";
import { cn } from "@/lib/utils";
import { Loader2 } from "lucide-react";

interface AppleButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: "primary" | "secondary" | "ghost" | "destructive" | "outline";
    size?: "sm" | "md" | "lg";
    isLoading?: boolean;
    asMotion?: boolean;
}

export const AppleButton = React.forwardRef<HTMLButtonElement, AppleButtonProps>(
    ({ className, variant = "primary", size = "md", isLoading, children, disabled, type = "button", ...props }, ref) => {
        const baseStyles = "inline-flex items-center justify-center font-medium transition-all focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50 disabled:pointer-events-none active:scale-[0.98]";

        const variants = {
            primary: "bg-primary text-primary-foreground hover:brightness-110 shadow-sm",
            secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
            ghost: "hover:bg-secondary/50 text-foreground",
            destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
            outline: "border border-input bg-background hover:bg-secondary/50 text-foreground"
        };

        const sizes = {
            sm: "h-8 px-3 text-xs rounded-full",
            md: "h-10 px-5 text-sm rounded-full",
            lg: "h-12 px-8 text-base rounded-full",
        };

        return (
            <button
                ref={ref}
                type={type}
                disabled={disabled || isLoading}
                className={cn(baseStyles, variants[variant], sizes[size], className)}
                {...props}
            >
                {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {children}
            </button>
        );
    }
);
AppleButton.displayName = "AppleButton";
