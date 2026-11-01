/**
 * Design System Tokens
 * Centralized design configuration for consistent UI/UX across all pages
 */

export const designTokens = {
    // Typography
    typography: {
        fontFamily: {
            sans: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            mono: 'ui-monospace, monospace',
        },
        fontSize: {
            xs: '0.75rem',      // 12px
            sm: '0.875rem',     // 14px
            base: '1rem',       // 16px
            lg: '1.125rem',     // 18px
            xl: '1.25rem',      // 20px
            '2xl': '1.5rem',    // 24px
            '3xl': '1.875rem',  // 30px
            '4xl': '2.25rem',   // 36px
            '5xl': '3rem',      // 48px
            '6xl': '3.75rem',   // 60px
        },
        fontWeight: {
            normal: '400',
            medium: '500',
            semibold: '600',
            bold: '700',
            black: '900',
        },
        lineHeight: {
            tight: '1.25',
            normal: '1.5',
            relaxed: '1.75',
        },
    },

    // Colors - Premium palette
    colors: {
        primary: {
            light: '#3b82f6',      // Blue
            DEFAULT: '#2563eb',
            dark: '#1e40af',
        },
        success: {
            light: '#10b981',      // Emerald
            DEFAULT: '#059669',
            dark: '#047857',
        },
        warning: {
            light: '#f59e0b',      // Amber
            DEFAULT: '#d97706',
            dark: '#b45309',
        },
        error: {
            light: '#ef4444',      // Rose
            DEFAULT: '#dc2626',
            dark: '#b91c1c',
        },
        neutral: {
            50: '#fafafa',
            100: '#f5f5f5',
            200: '#e5e5e5',
            300: '#d4d4d4',
            400: '#a3a3a3',
            500: '#737373',
            600: '#525252',
            700: '#404040',
            800: '#262626',
            900: '#171717',
        },
    },

    // Spacing - 8pt grid system
    spacing: {
        xs: '0.5rem',    // 8px
        sm: '0.75rem',   // 12px
        md: '1rem',      // 16px
        lg: '1.5rem',    // 24px
        xl: '2rem',      // 32px
        '2xl': '3rem',   // 48px
        '3xl': '4rem',   // 64px
        '4xl': '6rem',   // 96px
    },

    // Border radius
    borderRadius: {
        sm: '0.375rem',  // 6px
        md: '0.5rem',    // 8px
        lg: '0.75rem',   // 12px
        xl: '1rem',      // 16px
        '2xl': '1.5rem', // 24px
        full: '9999px',
    },

    // Shadows
    shadows: {
        sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
        md: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
        lg: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
        xl: '0 20px 25px -5px rgb(0 0 0 / 0.1)',
        inner: 'inset 0 2px 4px 0 rgb(0 0 0 / 0.05)',
    },

    // Animation durations
    animation: {
        fast: '150ms',
        base: '300ms',
        slow: '500ms',
        slower: '1000ms',
    },

    // Z-index layers
    zIndex: {
        base: 0,
        dropdown: 10,
        sticky: 20,
        modal: 30,
        popover: 40,
        toast: 50,
    },
} as const;

// Page layout constants
export const pageLayout = {
    maxWidth: '1400px',
    padding: {
        mobile: '1rem',
        tablet: '1.5rem',
        desktop: '2rem',
    },
    headerHeight: '5rem',  // 80px
    sectionSpacing: '4rem', // 64px
} as const;

// Common CSS classes for consistency
export const commonClasses = {
    // Page containers
    pageContainer: 'min-h-screen bg-gradient-to-b from-background via-background to-background/95',
    contentWrapper: 'container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-7xl',

    // Headings
    h1: 'text-4xl sm:text-5xl font-black tracking-tight',
    h2: 'text-3xl sm:text-4xl font-bold',
    h3: 'text-2xl sm:text-3xl font-bold',
    h4: 'text-xl sm:text-2xl font-semibold',
    h5: 'text-lg sm:text-xl font-semibold',

    // Body text
    bodyLarge: 'text-lg leading-relaxed',
    body: 'text-base leading-normal',
    bodySmall: 'text-sm leading-normal',

    // Card styles
    card: 'rounded-2xl border border-border bg-card/50 backdrop-blur-sm p-6 shadow-lg',
    cardInteractive: 'rounded-2xl border border-border bg-card/50 backdrop-blur-sm p-6 shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-[1.02]',

    // Button styles (consistent with AppleButton)
    buttonPrimary: 'px-6 py-3 rounded-xl font-semibold bg-primary text-primary-foreground hover:bg-primary/90 transition-all duration-200',
    buttonSecondary: 'px-6 py-3 rounded-xl font-semibold bg-secondary text-secondary-foreground hover:bg-secondary/80 transition-all duration-200',
    buttonGhost: 'px-6 py-3 rounded-xl font-semibold hover:bg-accent hover:text-accent-foreground transition-all duration-200',

    // Layout sections
    section: 'py-12 sm:py-16',
    sectionCentered: 'py-12 sm:py-16 text-center',

    // Grid layouts
    grid2: 'grid grid-cols-1 md:grid-cols-2 gap-6',
    grid3: 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6',
    grid4: 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6',

    // Flex layouts
    flexCenter: 'flex items-center justify-center',
    flexBetween: 'flex items-center justify-between',
    flexCol: 'flex flex-col gap-4',
} as const;

// Background patterns
export const backgroundPatterns = {
    gradient: 'bg-gradient-to-b from-background via-background to-background/95',
    dots: 'bg-[radial-gradient(circle_at_1px_1px,_rgb(var(--foreground)/0.05)_1px,_transparent_0)] bg-[size:24px_24px]',
    grid: 'bg-[linear-gradient(to_right,_rgb(var(--foreground)/0.05)_1px,_transparent_1px),_linear-gradient(to_bottom,_rgb(var(--foreground)/0.05)_1px,_transparent_1px)] bg-[size:24px_24px]',
} as const;

export default designTokens;
