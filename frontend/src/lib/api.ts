/**
 * Smart API URL Configuration
 * Works on localhost, GitHub Pages, and remaining Vercel hosts.
 *
 * Usage: import API_URL from '@/lib/api'
 * Then: `${API_URL}/endpoint` works everywhere!
 */

class SmartAPIURL {
    private baseUrl: string;
    private isProd: boolean;

    constructor() {
        this.isProd = this.checkProduction();
        this.baseUrl = this.getBaseUrl();
    }

    private checkProduction(): boolean {
        if (typeof window === 'undefined') {
            return process.env.NODE_ENV === 'production';
        }
        const hostname = window.location.hostname;
        // Exact domain matching to prevent subdomain bypass
        return hostname === 'vercel.app' ||
            hostname.endsWith('.vercel.app') ||
            hostname.endsWith('.github.io') ||
            hostname === 'yourdomain.com' ||
            hostname.endsWith('.yourdomain.com') ||
            process.env.NODE_ENV === 'production';
    }

    private isValidOrigin(url: string): boolean {
        // Security: Validate URL to prevent injection attacks
        try {
            const parsed = new URL(url);
            // Only allow http/https protocols
            if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
                return false;
            }
            // Block private/local IPs
            const hostname = parsed.hostname.toLowerCase();
            if (hostname === 'localhost' ||
                hostname === '127.0.0.1' ||
                hostname === '0.0.0.0' ||
                hostname.startsWith('10.') ||
                hostname.startsWith('192.168.') ||
                hostname.startsWith('172.')) {
                return false;
            }
            return true;
        } catch {
            return false;
        }
    }


    private getBaseUrl(): string {
        if (typeof window === 'undefined') {
            return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        }

        const hostname = window.location.hostname;

        if (this.isProd) {
            if (process.env.NEXT_PUBLIC_API_URL) {
                return process.env.NEXT_PUBLIC_API_URL;
            }
            // GitHub Pages hosts only the static frontend; API is not same-origin.
            if (hostname.endsWith('.github.io')) {
                return 'http://localhost:8000';
            }
            // Vercel (or other Node hosts): same origin, endpoints at /api/*
            const origin = window.location.origin;
            if (this.isValidOrigin(origin)) {
                return origin + '/api';
            }
            return '/api';
        }

        // Localhost
        if (hostname === 'localhost' || hostname === '127.0.0.1') {
            return 'http://localhost:8000';
        }

        return 'http://localhost:8000';
    }

    // Make the object behave like a string
    toString(): string {
        return this.baseUrl;
    }

    valueOf(): string {
        return this.baseUrl;
    }

    // Allow concatenation
    concat(...args: string[]): string {
        return this.baseUrl + args.join('');
    }
}

// Create instance
const apiUrlInstance = new SmartAPIURL();

// Export as string-like object
const API_URL = apiUrlInstance.toString();

export default API_URL;

// Export helper functions
export const isProduction = (): boolean => {
    if (typeof window === 'undefined') {
        return process.env.NODE_ENV === 'production';
    }
    const hostname = window.location.hostname;
    // Exact domain matching to prevent subdomain bypass
    return hostname === 'vercel.app' ||
        hostname.endsWith('.vercel.app') ||
        hostname.endsWith('.github.io') ||
        hostname === 'yourdomain.com' ||
        hostname.endsWith('.yourdomain.com');
};

export const getApiEndpoint = (path: string): string => {
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    return API_URL + cleanPath;
};
