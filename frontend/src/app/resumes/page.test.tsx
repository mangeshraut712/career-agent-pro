import type { AnchorHTMLAttributes, ReactNode } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ResumesPage from "./page";

const mocks = vi.hoisted(() => ({
  exportResumeDocx: vi.fn(),
  exportResumePdf: vi.fn(),
  toast: vi.fn(),
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
    currentJob: null,
    profile,
    updateProfile: vi.fn(),
  }),
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
});
