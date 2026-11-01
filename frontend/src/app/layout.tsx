import type { Metadata, Viewport } from "next";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { ToastProvider } from "@/components/ui/Toast";
import "./globals.css";

// Viewport configuration for mobile optimization
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#0a0a0a" },
  ],
};

export const metadata: Metadata = {
  metadataBase: new URL("https://mangeshraut712.github.io/career-agent-pro/"),
  title: "CareerAgentPro - AI-Powered Career Platform",
  description:
    "Your AI Career Co-Pilot. Automate job search, optimize resumes with AI, and land your dream role with CareerAgentPro.",
  keywords: [
    "AI resume",
    "job search",
    "career platform",
    "resume optimization",
    "job application",
  ],
  authors: [{ name: "Mangesh Raut" }],
  robots: {
    index: true,
    follow: true,
  },
  openGraph: {
    title: "CareerAgentPro - AI-Powered Career Platform",
    description:
      "Your AI Career Co-Pilot. Automate job search, optimize resumes, and land your dream role.",
    url: "https://mangeshraut712.github.io/career-agent-pro/",
    siteName: "CareerAgentPro",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "CareerAgentPro - AI Career Platform",
    description: "Your AI Career Co-Pilot for job search automation.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* DNS Prefetch for external resources */}
        <link rel="dns-prefetch" href="https://openrouter.ai" />
      </head>
      <body
        className="font-sans antialiased min-h-screen flex flex-col"
        suppressHydrationWarning
      >
        <ToastProvider>
          <Navbar />
          <main className="flex-1 pt-20">{children}</main>
          <Footer />
        </ToastProvider>
      </body>
    </html>
  );
}
