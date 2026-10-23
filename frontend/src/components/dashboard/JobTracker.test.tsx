import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { JobTracker } from "@/components/dashboard/JobTracker";

const job = {
    id: "job-1",
    title: "Platform Engineer",
    company: "Acme",
    url: "javascript:alert(1)",
    status: "applied" as const,
    createdAt: "2026-01-01T00:00:00Z",
};

describe("JobTracker accessibility", () => {
    it("names the status control, posting link, and remove button", () => {
        render(
            <JobTracker
                jobs={[job]}
                onUpdateStatus={vi.fn()}
                onDelete={vi.fn()}
            />
        );

        expect(screen.getByRole("combobox", { name: /application status for platform engineer/i })).toBeInTheDocument();
        const posting = screen.getByRole("link", { name: /open job posting for platform engineer/i });
        expect(posting).toHaveAttribute("href", "#");
        expect(screen.getByRole("button", { name: /remove platform engineer/i })).toBeInTheDocument();
    });
});
