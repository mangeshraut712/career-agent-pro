"""
Competency-Based Job Description Assessment Service

Analyzes job descriptions to extract competencies, assign weightage,
and calculate fit scores for better resume tailoring.
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import Counter


class CompetencyAssessor:
    """Analyzes JDs and extracts competencies with weightage"""
    
    # Pre-defined competency categories with keywords
    COMPETENCY_KEYWORDS = {
        "Product Strategy": [
            "product vision", "roadmap", "strategy", "prioritization",
            "product-market fit", "pmf", "market analysis", "competitive analysis",
            "go-to-market", "gtm", "positioning", "product launch"
        ],
        
        "Technical Skills": [
            "engineering", "technical", "architecture", "api", "backend",
            "frontend", "full-stack", "database", "cloud", "aws", "azure",
            "python", "javascript", "react", "node", "sql", "system design"
        ],
        
        "Data & Analytics": [
            "analytics", "metrics", "kpis", "data-driven", "ab testing",
            "sql", "tableau", "looker", "experiments", "analysis",
            "reporting", "insights", "business intelligence", "bi"
        ],
        
        "Stakeholder Management": [
            "stakeholder", "cross-functional", "alignment", "executive",
            "communication", "collaboration", "influence", "consensus",
            "presentations", "relationship building"
        ],
        
        "Leadership & Team": [
            "lead", "mentor", "coach", "team", "manage", "direct",
            "hire", "hiring", "people management", "leadership",
            "delegation", "motivation", "team building"
        ],
        
        "User Research & Design": [
            "user research", "ux", "ui", "usability", "user testing",
            "customer interviews", "personas", "journey mapping",
            "design thinking", "wireframes", "prototypes"
        ],
        
        "Execution & Delivery": [
            "execution", "delivery", "agile", "scrum", "sprint",
            "project management", "release", "launch", "ship",
            "implementation", "timelines", "dependencies"
        ],
        
        "Business Acumen": [
            "revenue", "profit", "roi", "business model", "pricing",
            "market", "customer acquisition", "retention", "growth",
            "business case", "p&l", "budget"
        ],
        
        "Communication": [
            "communication", "presentation", "documentation", "writing",
            "articulate", "storytelling", "influence", "negotiation",
            "public speaking", "executive communication"
        ],
        
        "Problem Solving": [
            "problem solving", "analytical", "critical thinking",
            "troubleshooting", "root cause", "creative", "innovative",
            "strategic thinking", "decision making"
        ]
    }
    
    # Company stage indicators
    STAGE_INDICATORS = {
        "early_stage": [
            "startup", "early-stage", "seed", "pre-seed", "series a",
            "mvp", "0-1", "founding", "launch", "bootstrap"
        ],
        "growth_stage": [
            "series b", "series c", "scaling", "growth", "expanding",
            "100+ employees", "rapid growth", "scale-up"
        ],
        "enterprise": [
            "fortune 500", "enterprise", "global", "multinational",
            "established", "1000+ employees", "mature", "public company"
        ]
    }
    
    @classmethod
    def assess_job_description(
        cls,
        jd_text: str,
        requirements: List[str] = None,
        skills: List[str] = None
    ) -> Dict:
        """
        Comprehensive JD assessment with competency breakdown.
        
        Args:
            jd_text: Full job description text
            requirements: List of requirements (optional)
            skills: List of required skills (optional)
            
        Returns:
            Dict with competencies, weightage, stage, and recommendations
        """
        # Combine all text safely
        full_text = (jd_text or "").lower()
        if requirements:
            full_text += " " + " ".join(requirements).lower()
        if skills:
            full_text += " " + " ".join(skills).lower()
        
        # Extract competencies with scores
        competencies = cls._extract_competencies(full_text)
        
        # Calculate weightage (percentage distribution)
        total_score = sum(c.get("score", 0) for c in competencies)
        for comp in competencies:
            comp["weightage"] = round((comp["score"] / total_score) * 100, 1) if total_score > 0 else 0.0
        
        # Detect company stage
        stage = cls._detect_company_stage(full_text)
        
        # Extract key requirements
        key_requirements = cls._extract_key_requirements(jd_text or "")
        
        # Generate recommendations
        recommendations = cls._generate_recommendations(competencies, stage)
        
        return {
            "competencies": competencies,
            "company_stage": stage,
            "key_requirements": key_requirements,
            "recommendations": recommendations,
            "total_competencies_found": len(competencies)
        }
    
    @classmethod
    def calculate_fit_score(
        cls,
        jd_assessment: Dict,
        user_skills: List[str],
        user_experience: List[Dict]
    ) -> Dict:
        """
        Calculate how well user fits the JD based on competencies.
        
        Args:
            jd_assessment: Result from assess_job_description
            user_skills: User's skill list
            user_experience: User's experience list
            
        Returns:
            Dict with fit scores per competency and overall
        """
        user_skills = user_skills or []
        user_experience = user_experience or []
        user_text = (" ".join(user_skills) + " " + " ".join([
            f"{exp.get('role', '')} {exp.get('description', '')}"
            for exp in user_experience
            if isinstance(exp, dict)
        ])).lower()
        
        competency_scores = []
        
        for comp in jd_assessment.get("competencies", []):
            # Count matching keywords
            keywords = cls.COMPETENCY_KEYWORDS.get(comp["name"], [])
            matches = sum(1 for kw in keywords if kw in user_text)
            total_keywords = len(keywords)
            
            if total_keywords > 0:
                score = min(100, int((matches / total_keywords) * 150))
            else:
                score = 0
            
            competency_scores.append({
                "competency": comp["name"],
                "weightage": comp.get("weightage", 0.0),
                "your_score": score,
                "matched_keywords": matches,
                "total_keywords": total_keywords,
                "gap": max(0, 100 - score)
            })
        
        # Calculate weighted overall fit
        if competency_scores:
            overall_fit = sum(
                cs["your_score"] * (cs["weightage"] / 100)
                for cs in competency_scores
            )
        else:
            overall_fit = 0
        
        return {
            "overall_fit": int(overall_fit),
            "competency_scores": competency_scores,
            "top_strengths": sorted(competency_scores, key=lambda x: x["your_score"], reverse=True)[:3],
            "top_gaps": sorted(competency_scores, key=lambda x: x["gap"], reverse=True)[:3]
        }
    
    @classmethod
    def _extract_competencies(cls, text: str) -> List[Dict]:
        """Extract competencies from text with scoring"""
        competency_scores = {}
        
        for competency, keywords in cls.COMPETENCY_KEYWORDS.items():
            score = 0
            found_keywords = []
            
            for keyword in keywords:
                # Count occurrences
                count = text.count(keyword)
                if count > 0:
                    score += count
                    found_keywords.append(keyword)
            
            if score > 0:
                competency_scores[competency] = {
                    "name": competency,
                    "score": score,
                    "keywords_found": found_keywords,
                    "keyword_count": len(found_keywords)
                }
        
        # Sort by score
        sorted_competencies = sorted(
            competency_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return sorted_competencies
    
    @classmethod
    def _detect_company_stage(cls, text: str) -> str:
        """Detect company stage from JD text"""
        stage_scores = {
            "early_stage": 0,
            "growth_stage": 0,
            "enterprise": 0
        }
        
        for stage, indicators in cls.STAGE_INDICATORS.items():
            for indicator in indicators:
                if indicator in text:
                    stage_scores[stage] += 1
        
        # Return stage with highest score
        if max(stage_scores.values()) == 0:
            return "unknown"
        
        return max(stage_scores, key=stage_scores.get)
    
    @classmethod
    def _extract_key_requirements(cls, jd_text: str) -> List[str]:
        """Extract key must-have requirements"""
        requirements = []
        
        # Look for "required", "must have", "essential" sections
        required_patterns = [
            r"(?:required|must have|essential)[\s:]+(.+?)(?:\n\n|\Z)",
            r"(?:minimum qualifications?)[\s:]+(.+?)(?:\n\n|\Z)",
            r"(?:you|candidate) must[\s:]+(.+?)(?:\n\n|\Z)"
        ]
        
        for pattern in required_patterns:
            matches = re.findall(pattern, jd_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Split by bullet points or newlines
                items = re.split(r'[•\-\*\n]+', match)
                for item in items:
                    cleaned = item.strip()
                    if len(cleaned) > 10 and len(cleaned) < 200:
                        requirements.append(cleaned)
        
        return requirements[:10]  # Limit to top 10
    
    @classmethod
    def _generate_recommendations(
        cls,
        competencies: List[Dict],
        stage: str
    ) -> List[str]:
        """Generate tailored recommendations based on assessment"""
        recommendations = []
        
        if not competencies:
            return ["Unable to extract competencies. Review JD manually."]
        
        # Top competency recommendation
        top_comp = competencies[0]
        recommendations.append(
            f"Emphasize {top_comp['name']} - it's heavily weighted in this role"
        )
        
        # Stage-specific recommendations
        if stage == "early_stage":
            recommendations.append(
                "Highlight speed of execution and ability to wear multiple hats"
            )
            recommendations.append(
                "Use action words: shipped, validated, launched, iterated"
            )
        elif stage == "growth_stage":
            recommendations.append(
                "Emphasize scalability and metrics-driven achievements"
            )
            recommendations.append(
                "Use action words: scaled, optimized, grew, increased by X%"
            )
        elif stage == "enterprise":
            recommendations.append(
                "Highlight stakeholder management and process excellence"
            )
            recommendations.append(
                "Use action words: coordinated, aligned, governed, ensured compliance"
            )
        
        # Competency coverage
        if len(competencies) > 5:
            recommendations.append(
                f"This role requires {len(competencies)} competencies - ensure broad skill coverage"
            )
        
        # Top keywords to include
        if competencies:
            top_keywords = []
            for comp in competencies[:3]:
                top_keywords.extend(comp["keywords_found"][:2])
            recommendations.append(
                f"Include these keywords: {', '.join(top_keywords[:5])}"
            )
        
        return recommendations


# Example usage
if __name__ == "__main__":
    sample_jd = """
    Senior Product Manager - Growth
    
    We're looking for a Senior PM to drive our growth initiatives. You'll work 
    cross-functionally with engineering, design, and data teams to build and 
    scale features that increase user acquisition and retention.
    
    Requirements:
    - 5+ years in product management
    - Strong analytical skills and experience with A/B testing
    - Excellent stakeholder management
    - Data-driven decision making with SQL proficiency
    - Experience shipping products at scale
    - Track record of driving measurable growth (10%+ improvements)
    
    Nice to have:
    - Experience in Series B/C startups
    - Background in marketplace or platform products
    """
    
    # Assess JD
    assessment = CompetencyAssessor.assess_job_description(sample_jd)
    
    print("=== JD Assessment ===")
    print(f"Company Stage: {assessment['company_stage']}")
    print(f"\nCompetencies Found: {assessment['total_competencies_found']}")
    
    for comp in assessment['competencies']:
        print(f"\n{comp['name']}: {comp['weightage']}%")
        print(f"  Keywords: {', '.join(comp['keywords_found'][:3])}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(assessment['recommendations'], 1):
        print(f"{i}. {rec}")
    
    # Calculate fit
    user_skills = ["product management", "sql", "analytics", "ab testing", "stakeholder management"]
    user_exp = [
        {"role": "Product Manager", "description": "Led growth initiatives, increased retention by 15%"}
    ]
    
    fit = CompetencyAssessor.calculate_fit_score(assessment, user_skills, user_exp)
    
    print(f"\n=== Fit Analysis ===")
    print(f"Overall Fit: {fit['overall_fit']}/100")
    
    print(f"\nTop Strengths:")
    for strength in fit['top_strengths']:
        print(f"  - {strength['competency']}: {strength['your_score']}/100")
    
    print(f"\nTop Gaps:")
    for gap in fit['top_gaps']:
        print(f"  - {gap['competency']}: {gap['gap']} points to improve")
