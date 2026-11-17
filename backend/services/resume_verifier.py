"""
Resume Verification Service

Comprehensive quality validation for entire resumes before export.
Ensures all bullets meet standards and generates quality reports.
"""

from typing import Dict, List, Optional
from schemas import ResumeData, SixPointBullet, BulletValidationResult
from services.bullet_validator import BulletValidator


class ResumeVerifier:
    """Validates entire resumes and generates quality reports"""
    
    # Quality thresholds
    MIN_QUALITY_SCORE = 70  # Minimum to pass verification
    RECOMMENDED_SCORE = 85  # Recommended quality level
    EXCELLENT_SCORE = 95    # Excellent quality
    
    @classmethod
    def verify_resume(
        cls,
        resume: Optional[ResumeData] = None,
        bullets: Optional[List[SixPointBullet]] = None,
        strict_mode: bool = False
    ) -> Dict:
        """
        Comprehensive resume verification.
        
        Args:
            resume: Resume data to verify
            bullets: Optional list of 6-point bullets
            strict_mode: If True, enforce stricter standards
            
        Returns:
            Dict with verification results and quality report
        """
        if resume is None:
            return {
                "overall_pass": False,
                "overall_quality_score": 0,
                "can_export": False,
                "bullet_validation": [],
                "profile_validation": {
                    "score": 0,
                    "errors": ["Resume data is required"],
                    "warnings": [],
                    "completeness": {
                        "percentage": 0,
                        "completed_fields": 0,
                        "total_fields": 9,
                        "missing_fields": ["name", "email", "phone", "linkedin", "summary", "experience", "education", "skills", "projects"]
                    }
                },
                "errors": ["Resume data is required"],
                "warnings": [],
                "suggestions": ["Provide resume profile data with name, email, and experience"],
                "quality_report": {
                    "overall_score": 0,
                    "quality_level": "Needs Improvement",
                    "quality_emoji": "❌",
                    "can_export": False,
                    "profile_score": 0,
                    "profile_completeness": 0,
                    "bullet_stats": None,
                    "summary": "Resume needs significant improvement before submission. Address errors and warnings."
                }
            }

        results = {
            "overall_pass": False,
            "overall_quality_score": 0,
            "can_export": False,
            "bullet_validation": [],
            "profile_validation": {},
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "quality_report": {}
        }
        
        # 1. Validate Profile Completeness
        profile_validation = cls._validate_profile(resume)
        results["profile_validation"] = profile_validation
        
        if profile_validation["errors"]:
            results["errors"].extend(profile_validation["errors"])
        if profile_validation["warnings"]:
            results["warnings"].extend(profile_validation["warnings"])
        
        # 2. Validate Bullets (if provided)
        bullet_scores = []
        if bullets:
            for i, bullet in enumerate(bullets):
                if bullet is None:
                    continue
                validation = BulletValidator.validate_bullet(bullet)
                
                results["bullet_validation"].append({
                    "bullet_index": i,
                    "is_valid": validation.is_valid,
                    "quality_score": validation.quality_score,
                    "character_count": validation.character_count,
                    "has_metrics": validation.has_metrics,
                    "errors": validation.errors,
                    "warnings": validation.warnings
                })
                
                bullet_scores.append(validation.quality_score)
                
                # Collect bullet-specific errors
                if not validation.is_valid:
                    results["errors"].append(
                        f"Bullet {i+1}: {', '.join(validation.errors)}"
                    )
        
        # 3. Calculate Overall Quality Score
        profile_score = profile_validation["score"]
        
        if bullet_scores:
            avg_bullet_score = sum(bullet_scores) / len(bullet_scores)
            # Weighted: 40% profile, 60% bullets
            overall_score = int(profile_score * 0.4 + avg_bullet_score * 0.6)
        else:
            overall_score = profile_score
        
        results["overall_quality_score"] = overall_score
        
        # 4. Determine Pass/Fail
        threshold = cls.MIN_QUALITY_SCORE if not strict_mode else cls.RECOMMENDED_SCORE
        results["overall_pass"] = overall_score >= threshold
        
        # 5. Export Decision
        # Can export if:
        # - Overall pass
        # - No critical errors
        # - All bullets valid (if bullets provided)
        critical_errors = [e for e in results["errors"] if "must" in e.lower()]
        all_bullets_valid = all(b["is_valid"] for b in results["bullet_validation"]) if bullets else True
        
        results["can_export"] = (
            results["overall_pass"] and
            len(critical_errors) == 0 and
            all_bullets_valid
        )
        
        # 6. Generate Quality Report
        results["quality_report"] = cls._generate_quality_report(
            overall_score=overall_score,
            profile_validation=profile_validation,
            bullet_validation=results["bullet_validation"],
            can_export=results["can_export"]
        )
        
        # 7. Generate Suggestions
        results["suggestions"] = cls._generate_suggestions(
            overall_score=overall_score,
            profile_validation=profile_validation,
            bullet_scores=bullet_scores
        )
        
        return results
    
    @classmethod
    def _validate_profile(cls, resume: ResumeData) -> Dict:
        """Validate profile completeness and quality"""
        errors = []
        warnings = []
        score = 100  # Start at 100, deduct points for issues
        
        # Required fields
        if not resume.name or len(resume.name) < 2:
            errors.append("Name is required")
            score -= 20
        
        if not resume.email or "@" not in resume.email:
            errors.append("Valid email is required")
            score -= 20
        
        # Recommended fields
        if not resume.phone:
            warnings.append("Phone number recommended")
            score -= 5
        
        if not resume.linkedin:
            warnings.append("LinkedIn profile recommended")
            score -= 5
        
        # Summary quality
        if not resume.summary:
            errors.append("Professional summary is required")
            score -= 15
        elif len(resume.summary) < 50:
            warnings.append("Summary is too brief (< 50 chars)")
            score -= 10
        elif len(resume.summary) > 600:
            warnings.append("Summary is too long (> 600 chars)")
            score -= 5
        
        # Experience
        if not resume.experience or len(resume.experience) == 0:
            errors.append("At least one work experience is required")
            score -= 25
        else:
            # Check experience quality
            for i, exp in enumerate(resume.experience):
                if not isinstance(exp, dict):
                    warnings.append(f"Experience {i+1}: Invalid experience record")
                    score -= 5
                    continue

                if not exp.get("company"):
                    warnings.append(f"Experience {i+1}: Missing company name")
                    score -= 3
                
                if not exp.get("role"):
                    warnings.append(f"Experience {i+1}: Missing role/title")
                    score -= 3
                
                desc = exp.get("description")
                if not desc or len(str(desc)) < 20:
                    warnings.append(f"Experience {i+1}: Description too brief")
                    score -= 5
        
        # Education
        if not resume.education or len(resume.education) == 0:
            warnings.append("Education information recommended")
            score -= 10
        
        # Skills
        if not resume.skills or len(resume.skills) < 3:
            warnings.append("At least 3 skills recommended")
            score -= 10
        elif len(resume.skills) < 5:
            warnings.append("5+ skills recommended for better ATS compatibility")
            score -= 5
        
        # Ensure score doesn't go below 0
        score = max(0, score)
        
        return {
            "score": score,
            "errors": errors,
            "warnings": warnings,
            "completeness": cls._calculate_completeness(resume)
        }
    
    @classmethod
    def _calculate_completeness(cls, resume: ResumeData) -> Dict:
        """Calculate profile completeness percentage"""
        fields = {
            "name": bool(resume.name),
            "email": bool(resume.email),
            "phone": bool(resume.phone),
            "linkedin": bool(resume.linkedin),
            "summary": bool(resume.summary and len(resume.summary) >= 50),
            "experience": bool(resume.experience and len(resume.experience) > 0),
            "education": bool(resume.education and len(resume.education) > 0),
            "skills": bool(resume.skills and len(resume.skills) >= 5),
            "projects": bool(resume.projects and len(resume.projects) > 0)
        }
        
        completed = sum(1 for v in fields.values() if v)
        total = len(fields)
        percentage = int((completed / total) * 100)
        
        return {
            "percentage": percentage,
            "completed_fields": completed,
            "total_fields": total,
            "missing_fields": [k for k, v in fields.items() if not v]
        }
    
    @classmethod
    def _generate_quality_report(
        cls,
        overall_score: int,
        profile_validation: Dict,
        bullet_validation: List[Dict],
        can_export: bool
    ) -> Dict:
        """Generate human-readable quality report"""
        
        # Determine quality level
        if overall_score >= cls.EXCELLENT_SCORE:
            quality_level = "Excellent"
            quality_emoji = "🌟"
        elif overall_score >= cls.RECOMMENDED_SCORE:
            quality_level = "Good"
            quality_emoji = "✅"
        elif overall_score >= cls.MIN_QUALITY_SCORE:
            quality_level = "Acceptable"
            quality_emoji = "⚠️"
        else:
            quality_level = "Needs Improvement"
            quality_emoji = "❌"
        
        # Bullet statistics
        if bullet_validation:
            bullet_stats = {
                "total_bullets": len(bullet_validation),
                "valid_bullets": sum(1 for b in bullet_validation if b["is_valid"]),
                "avg_quality": int(sum(b["quality_score"] for b in bullet_validation) / len(bullet_validation)),
                "bullets_with_metrics": sum(1 for b in bullet_validation if b["has_metrics"]),
                "character_count_range": {
                    "min": min(b["character_count"] for b in bullet_validation),
                    "max": max(b["character_count"] for b in bullet_validation),
                    "avg": int(sum(b["character_count"] for b in bullet_validation) / len(bullet_validation))
                }
            }
        else:
            bullet_stats = None
        
        return {
            "overall_score": overall_score,
            "quality_level": quality_level,
            "quality_emoji": quality_emoji,
            "can_export": can_export,
            "profile_score": profile_validation["score"],
            "profile_completeness": profile_validation["completeness"]["percentage"],
            "bullet_stats": bullet_stats,
            "summary": cls._generate_summary(overall_score, can_export)
        }
    
    @classmethod
    def _generate_summary(cls, score: int, can_export: bool) -> str:
        """Generate summary text for quality report"""
        if score >= cls.EXCELLENT_SCORE:
            return "Excellent resume quality! Ready for submission to top companies."
        elif score >= cls.RECOMMENDED_SCORE:
            return "Good resume quality. Minor improvements recommended for best results."
        elif score >= cls.MIN_QUALITY_SCORE:
            if can_export:
                return "Acceptable quality. Consider improvements before applying to competitive roles."
            else:
                return "Quality threshold met, but critical errors prevent export. Fix errors first."
        else:
            return "Resume needs significant improvement before submission. Address errors and warnings."
    
    @classmethod
    def _generate_suggestions(
        cls,
        overall_score: int,
        profile_validation: Dict,
        bullet_scores: List[int]
    ) -> List[str]:
        """Generate actionable improvement suggestions"""
        suggestions = []
        
        # Profile suggestions
        if profile_validation["completeness"]["percentage"] < 80:
            missing = profile_validation["completeness"]["missing_fields"]
            suggestions.append(
                f"Complete your profile: Add {', '.join(missing[:3])}"
            )
        
        # Bullet suggestions
        if bullet_scores:
            avg_bullet = sum(bullet_scores) / len(bullet_scores)
            
            if avg_bullet < 80:
                suggestions.append(
                    "Improve bullet quality: Ensure all bullets have metrics and strong action verbs"
                )
            
            low_quality_bullets = sum(1 for s in bullet_scores if s < 70)
            if low_quality_bullets > 0:
                suggestions.append(
                    f"{low_quality_bullets} bullet(s) below quality threshold - review and improve"
                )
        
        # Score-based suggestions
        if overall_score < cls.RECOMMENDED_SCORE:
            suggestions.append(
                f"Target score: {cls.RECOMMENDED_SCORE}+ for best results (current: {overall_score})"
            )
        
        # Always useful suggestions
        if overall_score < 90:
            suggestions.append("Review all bullets for 240-260 character count")
            suggestions.append("Ensure every bullet contains quantified metrics")
        
        return suggestions[:5]  # Limit to top 5


# Example usage
if __name__ == "__main__":
    from schemas import ResumeData, SixPointBullet
    
    # Sample resume
    resume = ResumeData(
        name="John Doe",
        email="john.doe@email.com",
        phone="555-1234",
        linkedin="linkedin.com/in/johndoe",
        summary="Product Manager with 5+ years experience driving growth",
        experience=[
            {
                "company": "Tech Corp",
                "role": "Senior PM",
                "duration": "2020-2024",
                "description": "Led product initiatives"
            }
        ],
        education=[
            {
                "institution": "University",
                "degree": "BS Computer Science",
                "graduation_year": "2019"
            }
        ],
        skills=["Product Management", "SQL", "Analytics", "Agile", "Stakeholder Management"]
    )
    
    # Sample bullets
    bullets = [
        SixPointBullet(
            action="Led",
            context="cross-functional team for payment platform",
            method="using Agile methodology",
            result="reducing processing time by 40%",
            impact="improving cash flow visibility",
            outcome="for Fortune 500 clients"
        )
    ]
    
    # Verify
    result = ResumeVerifier.verify_resume(resume, bullets)
    
    print("=== Resume Verification ===")
    print(f"Overall Score: {result['overall_quality_score']}/100")
    print(f"Can Export: {result['can_export']}")
    print(f"\nQuality Report:")
    report = result['quality_report']
    print(f"  Level: {report['quality_emoji']} {report['quality_level']}")
    print(f"  Profile: {report['profile_score']}/100")
    print(f"  Completeness: {report['profile_completeness']}%")
    print(f"\nSummary: {report['summary']}")
    
    if result['errors']:
        print(f"\nErrors:")
        for error in result['errors']:
            print(f"  ❌ {error}")
    
    if result['suggestions']:
        print(f"\nSuggestions:")
        for suggestion in result['suggestions']:
            print(f"  💡 {suggestion}")
