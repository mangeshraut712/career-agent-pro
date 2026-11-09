"""
Verification Service - Quality Gates for Resume and Cover Letter Generation.
Implements Apply-Pilot's verification gate system for quality outputs.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from services.bullet_framework import MetricDiversifier, ActionVerbChecker

# A metric in the summary: "40%", "$1,200" or "5k"/"3 M". The (?<!\d) makes a
# digit run start a match only at its first digit, so a long run of digits is
# scanned once rather than once per digit (CodeQL py/polynomial-redos).
_SUMMARY_METRIC_RE = re.compile(
    r"((?<!\d)\d+%|\$[\d,]+|(?<!\d)\d+\s*(?:k|m|b))", re.IGNORECASE
)

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class VerificationResult:
    name: str
    status: VerificationStatus
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class VerificationReport:
    document_type: str
    overall_status: VerificationStatus
    score: float
    checks: List[VerificationResult]
    suggestions: List[str]
    auto_retry_recommended: bool = False


class ResumeVerifier:
    """Resume Verification Gate."""
    
    REQUIRED_SECTIONS = ["summary", "experience", "education", "skills"]
    MIN_BULLET_CHARS = 240
    MAX_BULLET_CHARS = 260
    MIN_BULLETS_PER_ROLE = 2
    MAX_BULLETS_PER_ROLE = 5
    TARGET_TOTAL_BULLETS = 13
    MIN_SUMMARY_CHARS = 360
    MAX_SUMMARY_CHARS = 380
    
    @classmethod
    def verify(cls, resume_data: Dict[str, Any]) -> VerificationReport:
        checks = []
        checks.append(cls._verify_sections(resume_data))
        checks.append(cls._verify_contact(resume_data))
        checks.append(cls._verify_summary_length(resume_data))
        checks.append(cls._verify_summary_metrics(resume_data))
        checks.extend(cls._verify_bullets(resume_data))
        checks.append(cls._verify_bullet_counts(resume_data))
        checks.append(cls._verify_metric_diversity(resume_data))
        checks.append(cls._verify_action_verb_uniqueness(resume_data))
        checks.append(cls._verify_skills(resume_data))
        checks.append(cls._verify_bullet_symbols(resume_data))
        
        passed = sum(1 for c in checks if c.status == VerificationStatus.PASSED)
        score = (passed / len(checks)) * 100 if checks else 0
        
        failed_critical = any(c.status == VerificationStatus.FAILED for c in checks)
        overall_status = VerificationStatus.FAILED if failed_critical else (
            VerificationStatus.PASSED if score >= 80 else VerificationStatus.WARNING
        )
        
        suggestions = [f"Fix {c.name}: {c.message}" for c in checks if c.status != VerificationStatus.PASSED]
        
        return VerificationReport(
            document_type="resume", overall_status=overall_status, score=round(score, 1),
            checks=checks, suggestions=suggestions, auto_retry_recommended=failed_critical
        )
    
    @classmethod
    def _verify_sections(cls, data: Dict[str, Any]) -> VerificationResult:
        missing = [s for s in cls.REQUIRED_SECTIONS if s not in data or not data[s]]
        if not missing:
            return VerificationResult("sections", VerificationStatus.PASSED, "All required sections present")
        if missing == ["summary"]:
            return VerificationResult("sections", VerificationStatus.WARNING, "Summary missing")
        return VerificationResult("sections", VerificationStatus.FAILED, f"Missing: {', '.join(missing)}")
    
    @classmethod
    def _verify_contact(cls, data: Dict[str, Any]) -> VerificationResult:
        missing = [f for f in ["name", "email"] if f not in data or not data[f]]
        if not missing:
            return VerificationResult("contact", VerificationStatus.PASSED, "Contact info complete")
        return VerificationResult("contact", VerificationStatus.FAILED, f"Missing: {', '.join(missing)}")

    @classmethod
    def _verify_summary_length(cls, data: Dict[str, Any]) -> VerificationResult:
        summary = (data.get("summary") or "").strip()
        if not summary:
            return VerificationResult("summary_length", VerificationStatus.WARNING, "Summary missing")
        length = len(summary)
        if cls.MIN_SUMMARY_CHARS <= length <= cls.MAX_SUMMARY_CHARS:
            return VerificationResult("summary_length", VerificationStatus.PASSED, f"{length} chars")
        return VerificationResult(
            "summary_length",
            VerificationStatus.WARNING,
            f"{length} chars (target {cls.MIN_SUMMARY_CHARS}-{cls.MAX_SUMMARY_CHARS})"
        )

    @classmethod
    def _verify_summary_metrics(cls, data: Dict[str, Any]) -> VerificationResult:
        summary = (data.get("summary") or "").strip()
        if not summary:
            return VerificationResult("summary_metrics", VerificationStatus.WARNING, "Summary missing")
        has_metrics = bool(_SUMMARY_METRIC_RE.search(summary))
        if has_metrics:
            return VerificationResult(
                "summary_metrics",
                VerificationStatus.WARNING,
                "Remove metrics from summary (save for bullets)"
            )
        return VerificationResult("summary_metrics", VerificationStatus.PASSED, "No metrics in summary")
    
    @classmethod
    def _verify_bullets(cls, data: Dict[str, Any]) -> List[VerificationResult]:
        results = []
        for i, job in enumerate(data.get("experience", [])):
            bullets = cls._extract_bullets(job)
            if not bullets:
                results.append(VerificationResult(
                    f"bullets_{i}",
                    VerificationStatus.WARNING,
                    "0 bullets (target 2-5)"
                ))
                continue

            char_issues = [
                f"Bullet {j+1}: {len(b)} chars"
                for j, b in enumerate(bullets)
                if len(b) < cls.MIN_BULLET_CHARS or len(b) > cls.MAX_BULLET_CHARS
            ]

            status = VerificationStatus.WARNING if char_issues else VerificationStatus.PASSED
            results.append(VerificationResult(f"bullets_{i}", status, 
                f"{len(bullets)} bullets" + (f", {len(char_issues)} issues" if char_issues else "")))
        return results or [VerificationResult("bullets", VerificationStatus.WARNING, "No experience")]

    @classmethod
    def _verify_bullet_counts(cls, data: Dict[str, Any]) -> VerificationResult:
        issues = []
        total_bullets = 0
        for i, job in enumerate(data.get("experience", [])):
            bullets = cls._extract_bullets(job)
            total_bullets += len(bullets)
            if not (cls.MIN_BULLETS_PER_ROLE <= len(bullets) <= cls.MAX_BULLETS_PER_ROLE):
                issues.append(
                    f"Role {i + 1}: {len(bullets)} bullets (target {cls.MIN_BULLETS_PER_ROLE}-{cls.MAX_BULLETS_PER_ROLE})"
                )

        if total_bullets != cls.TARGET_TOTAL_BULLETS:
            issues.append(f"Total bullets: {total_bullets} (target {cls.TARGET_TOTAL_BULLETS})")

        if not issues:
            return VerificationResult("bullet_counts", VerificationStatus.PASSED, "Bullet counts in range")
        return VerificationResult("bullet_counts", VerificationStatus.WARNING, "; ".join(issues))

    @classmethod
    def _verify_metric_diversity(cls, data: Dict[str, Any]) -> VerificationResult:
        bullets = cls._collect_all_bullets(data)
        if not bullets:
            return VerificationResult("metric_diversity", VerificationStatus.WARNING, "No bullets to analyze")
        analysis = MetricDiversifier.check_diversity(bullets)
        if analysis.get("warnings"):
            return VerificationResult(
                "metric_diversity",
                VerificationStatus.WARNING,
                "; ".join(analysis["warnings"])
            )
        if not analysis.get("all_types_present"):
            return VerificationResult(
                "metric_diversity",
                VerificationStatus.WARNING,
                "Add more metric types for diversity"
            )
        return VerificationResult("metric_diversity", VerificationStatus.PASSED, "Metric diversity looks good")

    @classmethod
    def _verify_action_verb_uniqueness(cls, data: Dict[str, Any]) -> VerificationResult:
        bullets = cls._collect_all_bullets(data)
        if not bullets:
            return VerificationResult("action_verbs", VerificationStatus.WARNING, "No bullets to analyze")
        analysis = ActionVerbChecker.check_uniqueness(bullets)
        if not analysis.get("all_unique"):
            return VerificationResult(
                "action_verbs",
                VerificationStatus.WARNING,
                "Duplicate action verbs found"
            )
        return VerificationResult("action_verbs", VerificationStatus.PASSED, "Action verbs are unique")

    @staticmethod
    def _extract_bullets(job: Dict[str, Any]) -> List[str]:
        bullets = job.get("bullets", job.get("description", ""))
        if isinstance(bullets, list):
            return [b.strip() for b in bullets if isinstance(b, str) and b.strip()]
        if isinstance(bullets, str):
            return [b.strip() for b in bullets.split("\n") if b.strip()]
        return []

    @classmethod
    def _collect_all_bullets(cls, data: Dict[str, Any]) -> List[str]:
        all_bullets = []
        for job in data.get("experience", []):
            all_bullets.extend(cls._extract_bullets(job))
        return all_bullets
    
    @classmethod
    def _verify_skills(cls, data: Dict[str, Any]) -> VerificationResult:
        skills = data.get("skills", [])
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",")]
        if not skills:
            return VerificationResult("skills", VerificationStatus.FAILED, "No skills listed")
        
        # Apply-Pilot: No soft skills allowed in skills section
        soft_skills = [
            "stakeholder management", "leadership", "communication", 
            "collaboration", "teamwork", "problem solving", "scrappy",
            "fast-paced", "critical thinking", "gtm strategy"
        ]
        found_soft = [s for s in skills if s.lower() in soft_skills]
        
        if found_soft:
            return VerificationResult(
                "skills", 
                VerificationStatus.WARNING, 
                f"Remove soft skills: {', '.join(found_soft)}. Hard skills only."
            )
            
        if len(skills) < 5:
            return VerificationResult("skills", VerificationStatus.WARNING, f"Only {len(skills)} skills")
        return VerificationResult("skills", VerificationStatus.PASSED, f"{len(skills)} skills")

    @classmethod
    def _verify_bullet_symbols(cls, data: Dict[str, Any]) -> VerificationResult:
        """Verify bullets use current symbol (•) not dashes."""
        for job in data.get("experience", []):
            desc = job.get("description", job.get("bullets", ""))
            if isinstance(desc, str) and (desc.strip().startswith("-") or "\n-" in desc):
                return VerificationResult("bullet_symbols", VerificationStatus.WARNING, "Use '•' instead of '-' for bullets")
        return VerificationResult("bullet_symbols", VerificationStatus.PASSED, "Correct bullet symbols used")


class CoverLetterVerifier:
    """Cover Letter Verification Gate - 8-12 lines, 150-200 words, 4 paragraphs."""
    
    @classmethod
    def verify(cls, cover_letter: str, job_data: Optional[Dict] = None) -> VerificationReport:
        checks = []
        lines = [l for l in cover_letter.strip().split("\n") if l.strip()]
        words = cover_letter.split()
        paragraphs = [p for p in cover_letter.split("\n\n") if p.strip()]
        
        # Line count
        if 8 <= len(lines) <= 12:
            checks.append(VerificationResult("lines", VerificationStatus.PASSED, f"{len(lines)} lines"))
        else:
            checks.append(VerificationResult("lines", VerificationStatus.WARNING, f"{len(lines)} lines (target 8-12)"))
        
        # Word count
        if 150 <= len(words) <= 200:
            checks.append(VerificationResult("words", VerificationStatus.PASSED, f"{len(words)} words"))
        else:
            checks.append(VerificationResult("words", VerificationStatus.WARNING, f"{len(words)} words (target 150-200)"))

        # Paragraph structure
        if len(paragraphs) == 4:
            checks.append(VerificationResult("paragraphs", VerificationStatus.PASSED, "4 paragraphs"))
        else:
            checks.append(VerificationResult("paragraphs", VerificationStatus.WARNING, f"{len(paragraphs)} paragraphs (target 4)"))
        
        # Personalization
        if job_data:
            cl_lower = cover_letter.lower()
            company = job_data.get("company", "").lower()
            has_company = company and company in cl_lower
            if has_company:
                checks.append(VerificationResult("personalization", VerificationStatus.PASSED, "Mentions company"))
            else:
                checks.append(VerificationResult("personalization", VerificationStatus.WARNING, "Add company mention"))

            title = job_data.get("title", "").lower()
            first_paragraph = paragraphs[0].lower() if paragraphs else ""
            if title and title in first_paragraph:
                checks.append(VerificationResult("hook", VerificationStatus.PASSED, "Role mentioned in hook"))
            else:
                checks.append(VerificationResult("hook", VerificationStatus.WARNING, "Add role mention in opening"))

        # CTA check
        cta_phrases = [
            "would you be open", "could we", "happy to chat", "let me know",
            "i'd welcome", "i would welcome", "open to a quick", "quick call"
        ]
        has_cta = any(p in cover_letter.lower() for p in cta_phrases)
        if has_cta:
            checks.append(VerificationResult("cta", VerificationStatus.PASSED, "Has call-to-action"))
        else:
            checks.append(VerificationResult("cta", VerificationStatus.WARNING, "Add a call-to-action"))
        
        passed = sum(1 for c in checks if c.status == VerificationStatus.PASSED)
        score = (passed / len(checks)) * 100 if checks else 0
        overall = VerificationStatus.PASSED if score >= 80 else VerificationStatus.WARNING
        
        return VerificationReport(
            document_type="cover_letter", overall_status=overall, score=round(score, 1),
            checks=checks, suggestions=[c.message for c in checks if c.status != VerificationStatus.PASSED]
        )


class OutreachVerifier:
    """
    Outreach Message Verification Gate.
    Based on Aakash Gupta's 3-C's Framework: Customized, Compelling, Concise.
    """
    
    # Character count targets from Apply-Pilot
    TARGETS = {
        "tier_1": (600, 800),
        "tier_2": (400, 600),
        "tier_3": (300, 500)
    }
    
    @classmethod
    def verify(cls, message: str, tier: str, job_data: Optional[Dict] = None) -> VerificationReport:
        checks = []
        char_count = len(message)
        min_c, max_c = cls.TARGETS.get(tier, (400, 800))
        
        # 1. CONCISE Check (Length)
        if min_c <= char_count <= max_c:
            checks.append(VerificationResult("concise", VerificationStatus.PASSED, f"{char_count} chars (target {min_c}-{max_c})"))
        else:
            checks.append(VerificationResult("concise", VerificationStatus.WARNING, f"{char_count} chars (target {min_c}-{max_c})"))
        
        # 2. CUSTOMIZED Check (Personalization)
        cl_lower = message.lower()
        if job_data:
            company = job_data.get("company", "").lower()
            if company and company in cl_lower:
                checks.append(VerificationResult("customized", VerificationStatus.PASSED, "Mentions company"))
            else:
                checks.append(VerificationResult("customized", VerificationStatus.WARNING, "Missing company personalization"))
        
        # 3. COMPELLING Check (Value/CTA)
        cta_phrases = ["would you be open", "could we", "happy to chat", "let me know", "would appreciate", "guidance"]
        has_cta = any(p in cl_lower for p in cta_phrases)
        if has_cta:
            checks.append(VerificationResult("compelling", VerificationStatus.PASSED, "Clear call-to-action found"))
        else:
            checks.append(VerificationResult("compelling", VerificationStatus.WARNING, "Add more compelling ask/CTA"))
            
        passed = sum(1 for c in checks if c.status == VerificationStatus.PASSED)
        score = (passed / len(checks)) * 100 if checks else 0
        overall = VerificationStatus.PASSED if score >= 80 else VerificationStatus.WARNING
        
        return VerificationReport(
            document_type=f"outreach_{tier}", 
            overall_status=overall,
            score=round(score, 1), 
            checks=checks, 
            suggestions=[c.message for c in checks if c.status != VerificationStatus.PASSED]
        )
