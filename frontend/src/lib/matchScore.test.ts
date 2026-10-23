import { describe, expect, it } from "vitest";
import { calculateMatchScore, skillsMatch } from "@/lib/matchScore";

describe("skill match score", () => {
    it("does not treat Java as JavaScript or C as a substring", () => {
        expect(skillsMatch("Java", "JavaScript")).toBe(false);
        expect(skillsMatch("C", "React")).toBe(false);
        expect(skillsMatch(" Python ", "python")).toBe(true);
    });

    it("scores exact overlaps only", () => {
        const score = calculateMatchScore(
            ["JavaScript", "Python", "SQL"],
            ["Java", "Python"]
        );
        expect(score).toBe(33);
        expect(calculateMatchScore([], ["Python"])).toBe(0);
        expect(calculateMatchScore(["Python"], [])).toBe(0);
    });
});
