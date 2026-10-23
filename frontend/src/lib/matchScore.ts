/**
 * Exact skill comparison. Substring matching treats "Java" as "JavaScript"
 * and a one-letter skill such as "C" as a match for almost every word.
 */
export function skillsMatch(userSkill: string, jobSkill: string): boolean {
    const left = userSkill.trim().toLowerCase();
    const right = jobSkill.trim().toLowerCase();
    if (!left || !right) {
        return false;
    }
    return left === right;
}

export function calculateMatchScore(jobSkills: string[], userSkills: string[]): number {
    if (!jobSkills?.length || !userSkills?.length) {
        return 0;
    }
    const matched = jobSkills.filter((skill) =>
        userSkills.some((userSkill) => skillsMatch(userSkill, skill))
    );
    return Math.round((matched.length / jobSkills.length) * 100);
}
