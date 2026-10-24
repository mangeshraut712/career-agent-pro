"""
Unit Tests for JD Assessor Service
Tests the core functionality of job description assessment and fit scoring.
"""

import pytest
from services.jd_assessor import JDAssessor, FitLevel, assess_job_fit


class TestJDAssessor:
    """Test suite for JD Assessor service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_jd = """
        We are seeking a Senior Product Manager with 5+ years of experience.
        
        Requirements:
        - Strong technical skills in Python, SQL, and API design
        - Leadership experience managing cross-functional teams
        - Product strategy and roadmap planning
        - Excellent communication and stakeholder management
        - Experience with agile methodologies
        """
        
        self.sample_resume = {
            "name": "John Doe",
            "email": "john@example.com",
            "summary": "Product Manager with 6 years of experience",
            "experience": [
                {
                    "role": "Senior Product Manager",
                    "company": "TechCorp",
                    "duration": "2020-Present",
                    "bullets": [
                        "Led cross-functional team of 10 engineers",
                        "Built analytics platform using Python and SQL",
                        "Managed product roadmap and strategy"
                    ]
                }
            ],
            "skills": ["Python", "SQL", "Product Strategy", "Agile", "API Design"],
            "education": [{"institution": "University", "degree": "BS CS"}]
        }

    def test_assess_basic_fit(self):
        """Test basic JD assessment."""
        assessment = JDAssessor.assess(self.sample_jd, self.sample_resume)
        
        assert assessment.fit_score >= 0
        assert assessment.fit_score <= 100
        assert isinstance(assessment.fit_level, FitLevel)
        assert len(assessment.competency_matches) > 0

    def test_fit_score_calculation(self):
        """Test fit score is within expected range for good match."""
        assessment = JDAssessor.assess(self.sample_jd, self.sample_resume)
        
        # Should be a valid score based on weighted matching
        # The algorithm uses weighted competency scores, so exact score may vary
        assert assessment.fit_score >= 15, f"Expected fit score >= 15, got {assessment.fit_score}"
        assert assessment.fit_score <= 100, f"Expected fit score <= 100, got {assessment.fit_score}"
        
        # Check that it identified strengths correctly
        assert len(assessment.strengths) > 0, "Should have identified at least one strength"

    def test_competency_matching(self):
        """Test competency area matching."""
        assessment = JDAssessor.assess(self.sample_jd, self.sample_resume)
        
        # Check that competencies were identified
        assert any(m.name == "technical_skills" and m.match_score > 0 
                  for m in assessment.competency_matches)
        assert any(m.name == "leadership" and m.match_score > 0 
                  for m in assessment.competency_matches)

    def test_skills_intelligence(self):
        """Test skills intelligence analysis."""
        jd_with_tools = "Looking for PM with Tableau experience and SQL skills"
        assessment = JDAssessor.assess(jd_with_tools, self.sample_resume)
        
        assert "skills_intelligence" in dir(assessment)
        assert "prioritized_tier_1" in assessment.skills_intelligence
        assert "tier_2_swaps" in assessment.skills_intelligence

    def test_summary_guidance(self):
        """Test summary guidance generation."""
        assessment = JDAssessor.assess(self.sample_jd, self.sample_resume)
        
        assert assessment.summary_guidance
        assert "360-380 characters" in assessment.summary_guidance
        assert "No metrics" in assessment.summary_guidance or "Avoid metrics" in assessment.summary_guidance

    def test_interest_level_calculation(self):
        """Test interest level scoring."""
        assessment = JDAssessor.assess(self.sample_jd, self.sample_resume)
        
        assert assessment.interest_level >= 1
        assert assessment.interest_level <= 10

    def test_strategic_decision(self):
        """Test strategic decision logic."""
        assessment = JDAssessor.assess(self.sample_jd, self.sample_resume)
        
        assert assessment.decision in [
            "PROCEED (High Priority)",
            "PRESENT TO USER (Medium Priority)",
            "ARCHIVE (Low Priority)"
        ]

    def test_api_helper_function(self):
        """Test the assess_job_fit convenience function."""
        result = assess_job_fit(self.sample_jd, self.sample_resume)
        
        assert "fit_score" in result
        assert "fit_level" in result
        assert "strengths" in result
        assert "gaps" in result
        assert "skills_intelligence" in result
        assert "summary_guidance" in result
        assert "interest_level" in result
        assert "decision" in result

    def test_empty_resume(self):
        """Test handling of empty resume data."""
        empty_resume = {"name": "Test", "email": "test@test.com"}
        assessment = JDAssessor.assess(self.sample_jd, empty_resume)
        
        # Should not crash, but should have low score
        assert assessment.fit_score < 50

    def test_spinning_recommendation(self):
        """Test spinning recommendation generation."""
        startup_jd = "Early stage startup looking for scrappy PM who can wear many hats"
        assessment = JDAssessor.assess(startup_jd, self.sample_resume)
        
        assert assessment.spinning_recommendation
        assert any(keyword in assessment.spinning_recommendation.lower() 
                  for keyword in ["startup", "agile", "mvp", "early"])

    def test_keywords_do_not_match_inside_longer_words(self):
        """Java is not JavaScript, and ship is not hidden inside relationship."""
        jd = "JavaScript engineers maintain customer relationships."
        resume = {
            "skills": ["Java"],
            "experience": [{"role": "Engineer", "description": "Built Java services"}],
        }
        assessment = JDAssessor.assess(jd, resume)
        technical = next(m for m in assessment.competency_matches if m.name == "technical_skills")
        execution = next(m for m in assessment.competency_matches if m.name == "execution")
        mentioned = [kw.lower() for kw in technical.matched_keywords + technical.missing_keywords]
        assert "javascript" in mentioned
        assert "java" not in mentioned
        assert "ship" not in [kw.lower() for kw in execution.matched_keywords + execution.missing_keywords]

    def test_fit_score_is_normalized_to_mentioned_competencies(self):
        assessment = JDAssessor.assess(
            "Python developer",
            {"skills": ["Python"], "experience": []},
        )
        assert assessment.fit_score >= 80
        assert assessment.fit_level == FitLevel.EXCELLENT

    def test_decision_archives_weak_fit_even_when_interest_is_high(self):
        assert JDAssessor._determine_decision(40, 9) == "ARCHIVE (Low Priority)"
        assert JDAssessor._determine_decision(90, 5) == "ARCHIVE (Low Priority)"
        assert JDAssessor._determine_decision(80, 7) == "PRESENT TO USER (Medium Priority)"
        assert JDAssessor._determine_decision(90, 8) == "PROCEED (High Priority)"

    def test_interest_starts_below_the_present_threshold(self):
        assert JDAssessor._calculate_interest_level("a quiet office role", {}) == 5
        assert JDAssessor._calculate_interest_level("remote leadership role", {}) == 7
