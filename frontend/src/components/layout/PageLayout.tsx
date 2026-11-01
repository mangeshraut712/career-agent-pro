/**
 * PageLayout - Consistent layout wrapper for all pages
 * Ensures uniform spacing, background, and structure
 */

import { ReactNode } from "react";
import { motion } from "framer-motion";
import { FADE_IN } from "@/lib/animations";
import { commonClasses } from "@/lib/design-tokens";
import { cn } from "@/lib/utils";

interface PageLayoutProps {
  children: ReactNode;
  title?: string;
  description?: string;
  maxWidth?: "default" | "wide" | "full";
  className?: string;
  showHeader?: boolean;
}

export function PageLayout({
  children,
  title,
  description,
  maxWidth = "default",
  className,
  showHeader = true,
}: PageLayoutProps) {
  const maxWidthClasses = {
    default: "max-w-7xl",
    wide: "max-w-[1600px]",
    full: "max-w-full",
  };

  return (
    <div className={cn(commonClasses.pageContainer, className)}>
      <div
        className={cn(
          "container mx-auto px-4 sm:px-6 lg:px-8 py-8",
          maxWidthClasses[maxWidth],
        )}
      >
        {showHeader && (title || description) && (
          <motion.div {...FADE_IN} className="mb-8 text-center">
            {title && <h1 className={cn(commonClasses.h1, "mb-3")}>{title}</h1>}
            {description && (
              <p
                className={cn(
                  commonClasses.bodyLarge,
                  "text-muted-foreground max-w-2xl mx-auto",
                )}
              >
                {description}
              </p>
            )}
          </motion.div>
        )}

        {children}
      </div>
    </div>
  );
}

/**
 * PageSection - Consistent section wrapper
 */
interface PageSectionProps {
  children: ReactNode;
  title?: string;
  description?: string;
  className?: string;
  centered?: boolean;
}

export function PageSection({
  children,
  title,
  description,
  className,
  centered = false,
}: PageSectionProps) {
  return (
    <section
      className={cn(
        centered ? commonClasses.sectionCentered : commonClasses.section,
        className,
      )}
    >
      {(title || description) && (
        <div className={cn("mb-8", centered && "text-center")}>
          {title && <h2 className={cn(commonClasses.h2, "mb-3")}>{title}</h2>}
          {description && (
            <p className={cn(commonClasses.body, "text-muted-foreground")}>
              {description}
            </p>
          )}
        </div>
      )}
      {children}
    </section>
  );
}

/**
 * GridLayout - Consistent grid layouts
 */
interface GridLayoutProps {
  children: ReactNode;
  cols?: 2 | 3 | 4;
  className?: string;
}

export function GridLayout({ children, cols = 3, className }: GridLayoutProps) {
  const gridClasses = {
    2: commonClasses.grid2,
    3: commonClasses.grid3,
    4: commonClasses.grid4,
  };

  return <div className={cn(gridClasses[cols], className)}>{children}</div>;
}
