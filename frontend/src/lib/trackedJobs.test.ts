import { describe, expect, it } from "vitest";
import {
    removeStoredJob,
    toTrackedJob,
    updateStoredJobStatus,
    withStableIds,
} from "@/lib/trackedJobs";

const saved = {
    id: "job-1",
    url: "https://www.linkedin.com/jobs/view/1",
    title: "Platform Engineer",
    company: "Acme",
    description: "Build APIs",
    skills: ["Python", "Go"],
    matchScore: 80,
    analyzedAt: "2026-01-01T00:00:00.000Z",
    status: "applied",
};

describe("tracked job storage", () => {
    it("keeps a saved status instead of resetting every job to analyzing", () => {
        const view = toTrackedJob(saved, 0);
        expect(view.status).toBe("applied");
        expect(view.createdAt).toBe("2026-01-01T00:00:00.000Z");
    });

    it("preserves the full analysis when status changes or a job is removed", () => {
        const updated = updateStoredJobStatus([saved], "job-1", "interviewing");
        expect(updated[0].status).toBe("interviewing");
        expect(updated[0].description).toBe("Build APIs");
        expect(updated[0].skills).toEqual(["Python", "Go"]);
        expect(updated[0].matchScore).toBe(80);

        const removed = removeStoredJob(
            [saved, { ...saved, id: "job-2", title: "Other" }],
            "job-1"
        );
        expect(removed).toEqual([{ ...saved, id: "job-2", title: "Other" }]);
    });

    it("ignores an unknown status and keeps a stable id that can be updated", () => {
        const stored = withStableIds([{ title: "Role", status: "ghost", description: "Keep me" }]);
        expect(stored[0].id).toBe("job-0");
        expect(stored[0].description).toBe("Keep me");
        const view = toTrackedJob(stored[0], 0);
        expect(view.status).toBe("analyzing");
        const updated = updateStoredJobStatus(stored, view.id, "offer");
        expect(updated[0].status).toBe("offer");
        expect(updated[0].description).toBe("Keep me");
    });
});
