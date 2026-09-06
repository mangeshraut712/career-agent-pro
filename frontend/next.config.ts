import type { NextConfig } from "next";

const repoBasePath = "/career-agent-pro";
const isProd = process.env.NODE_ENV === "production";

const nextConfig: NextConfig = {
  // Static HTML export for GitHub Pages (no Node server).
  output: "export",
  // Project site lives at https://<user>.github.io/career-agent-pro/
  basePath: isProd ? repoBasePath : "",
  trailingSlash: true,

  // Enable React strict mode for better development experience
  reactStrictMode: true,

  // Disable x-powered-by header for security
  poweredByHeader: false,

  // Enable compression
  compress: true,

  // Image optimization settings (unoptimized: required for static export)
  images: {
    unoptimized: true,
    remotePatterns: [
      {
        protocol: "https",
        hostname: "**",
      },
    ],
    // Use modern image formats
    formats: ["image/avif", "image/webp"],
    // Optimize image loading
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048],
    imageSizes: [16, 32, 48, 64, 96, 128, 256],
    // Minimize image processing time
    minimumCacheTTL: 60 * 60 * 24 * 30, // 30 days
  },

  // Experimental performance features
  experimental: {
    // Optimize package imports for smaller bundles
    optimizePackageImports: [
      "lucide-react",
      "framer-motion",
      "date-fns",
      "zod",
      "react-hook-form",
    ],
  },
};

export default nextConfig;
