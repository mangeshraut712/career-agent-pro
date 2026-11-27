import type { AnchorHTMLAttributes, ReactNode } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ResumesPage from "./page";

const mocks = vi.hoisted(() => ({
  exportResumeDocx: vi.fn(),
  exportResumePdf: vi.fn(),
  toast: vi.fn(),
  post: vi.fn(),
  currentJob: null as null | {
    title: string;
    company: string;
    description: string;
    requirements: string[];
    skills: string[];
  },
}));

const profile = {
  name: "Ada Lovelace",
  email: "ada@example.com",
  phone: "555-0100",
  summary: "Built analytical engines.",
  skills: ["Python"],
  experience: [],
  education: [],
};

vi.mock("@/lib/resumeExport", () => ({
  exportResumeDocx: mocks.exportResumeDocx,
  exportResumePdf: mocks.exportResumePdf,
}));

vi.mock("@/hooks/useAppData", () => ({
  useAppData: () => ({
    currentJob: mocks.currentJob,
    profile,
    updateProfile: vi.fn(),
  }),
}));

vi.mock("axios", () => ({
  default: {
    post: (...args: unknown[]) => mocks.post(...args),
  },
}));

vi.mock("@/components/ui/Toast", () => ({
  useToast: () => ({ toast: mocks.toast }),
}));

vi.mock("next/link", () => ({
  default: ({
    children,
    href,
    ...props
  }: {
    children: ReactNode;
    href: string;
  } & AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

describe("ResumesPage exports", () => {
  beforeEach(() => {
    mocks.exportResumeDocx.mockReset().mockResolvedValue(undefined);
    mocks.exportResumePdf.mockReset().mockResolvedValue(undefined);
    mocks.toast.mockReset();
    mocks.post.mockReset();
    mocks.currentJob = null;
    profile.skills = ["Python"];
  });

  it("uses the PDF export client instead of only showing a success message", async () => {
    render(<ResumesPage />);

    fireEvent.click(screen.getByRole("button", { name: /render final pdf/i }));

    await waitFor(() => {
      expect(mocks.exportResumePdf).toHaveBeenCalledWith(
        expect.any(String),
        profile,
      );
    });
    expect(mocks.toast).toHaveBeenCalledWith(
      expect.stringMatching(/downloaded/i),
      "success",
    );
  });

  it("wires the DOCX control to the export client", async () => {
    render(<ResumesPage />);

    fireEvent.click(screen.getByRole("button", { name: /export as docx/i }));

    await waitFor(() => {
      expect(mocks.exportResumeDocx).toHaveBeenCalledWith(
        expect.any(String),
        profile,
      );
    });
    expect(mocks.toast).toHaveBeenCalledWith(
      expect.stringMatching(/downloaded/i),
      "success",
    );
  });

  it("prevents a second export while the first is still running", async () => {
    let finishExport: (() => void) | undefined;
    mocks.exportResumePdf.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          finishExport = resolve;
        }),
    );
    render(<ResumesPage />);

    const button = screen.getByRole("button", { name: /render final pdf/i });
    fireEvent.click(button);
    await waitFor(() => expect(mocks.exportResumePdf).toHaveBeenCalledOnce());

    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(mocks.exportResumePdf).toHaveBeenCalledOnce();

    finishExport?.();
    await waitFor(() =>
      expect(mocks.toast).toHaveBeenCalledWith(
        expect.stringMatching(/downloaded/i),
        "success",
      ),
    );
  });

  it("does not count Java as JavaScript when the API is unavailable", async () => {
    profile.skills = ["Java"];
    mocks.currentJob = {
      title: "Frontend Engineer",
      company: "Example",
      description: "Build interfaces.",
      requirements: [],
      skills: ["JavaScript", "Python"],
    };
    mocks.post.mockRejectedValue(new Error("offline"));

    render(<ResumesPage />);
    fireEvent.click(screen.getByRole("button", { name: /initiate deep forge/i }));

    expect(await screen.findByText(/Verified Skills \(0\)/)).toBeInTheDocument();
    expect(screen.getByText(/Optimization Req\. \(2\)/)).toBeInTheDocument();
    expect(screen.getByText("JavaScript")).toBeInTheDocument();
  });
});
