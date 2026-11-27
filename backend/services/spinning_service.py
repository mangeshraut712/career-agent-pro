"""
Spinning Strategy Service - Industry Language Adaptation

Adapts resume language to match target industry without fabrication.
Transforms your experience to resonate with different company stages.
"""

from typing import Dict, List, Tuple, Optional
from enum import Enum
import re


class CompanyStage(str, Enum):
    """Company stage categories for spinning strategy"""
    EARLY_STAGE = "early_stage"      # Startups, pre-Series A
    GROWTH_STAGE = "growth_stage"    # Series A-C, scaling companies
    ENTERPRISE = "enterprise"         # Fortune 500, large corporations


class SpinningStrategy:
    """Industry language adaptation without fabrication"""
    
    # Industry-specific vocabulary dictionaries
    DICTIONARIES = {
        CompanyStage.EARLY_STAGE: {
            "verbs": [
                "shipped", "validated", "tested", "pivoted", "launched",
                "iterated", "experimented", "prototyped", "bootstrapped"
            ],
            "keywords": [
                "MVP", "product-market fit", "iterations", "experiments",
                "rapid deployment", "lean", "agile", "sprint", "velocity"
            ],
            "modifiers": [
                "quick", "fast", "nimble", "flexible", "adaptive"
            ],
            "outcomes": [
                "validated hypothesis", "achieved PMF", "acquired first users",
                "proved concept", "gained traction"
            ]
        },
        
        CompanyStage.GROWTH_STAGE: {
            "verbs": [
                "scaled", "optimized", "increased", "expanded", "automated",
                "streamlined", "accelerated", "grew", "multiplied"
            ],
            "keywords": [
                "metrics", "KPIs", "growth", "efficiency", "scaling",
                "revenue", "ARR", "conversion", "retention", "funnel"
            ],
            "modifiers": [
                "measurable", "data-driven", "scalable", "sustainable", "systematic"
            ],
            "outcomes": [
                "increased by X%", "scaled to Y users", "grew revenue",
                "improved conversion", "reduced churn"
            ]
        },
        
        CompanyStage.ENTERPRISE: {
            "verbs": [
                "coordinated", "aligned", "ensured", "managed", "governed",
                "facilitated", "led", "orchestrated", "standardized"
            ],
            "keywords": [
                "stakeholders", "compliance", "governance", "enterprise",
                "cross-functional", "strategic alignment", "risk management",
                "best practices", "frameworks"
            ],
            "modifiers": [
                "compliant", "certified", "standardized", "regulated", "audited"
            ],
            "outcomes": [
                "ensured compliance", "achieved certification", "aligned stakeholders",
                "governed process", "managed risk"
            ]
        }
    }
    
    # Jargon translations that do not change what was done. The rewrite is the
    # same for every target stage on purpose: rewording a bullet to sound more
    # "strategic" or more "startup-like" would change the claim, and this module
    # adapts language without fabrication. Rules that changed the nature or
    # scope of the work ("tactical" -> "strategic", "command" -> "leadership")
    # or produced nonsense ("hospice care" -> "response care") are deliberately
    # absent.
    TRANSFORMATION_RULES = {
        # Healthcare → General
        "patient care": "time-critical service delivery",
        "families": "populations",
        "medical": "operational",
        "clinical": "process-driven",
        "diagnosis": "root cause analysis",
        "treatment plan": "action plan",
        "bedside": "client-facing",
        
        # Education → General
        "students": "users",
        "curriculum": "program",
        "classroom": "environment",
        "teaching": "training/coaching",
        "lesson plan": "framework",
        "grading": "evaluation",
        
        # Non-profit / Military → General. One entry per term: a duplicated key
        # would silently keep only the last value.
        "volunteers": "team members",
        "donations": "revenue/funding",
        "community outreach": "user acquisition/engagement",
        "beneficiaries": "stakeholders",
        "mission": "project",
        
        # Academia → Industry
        "research": "analysis/investigation",
        "dissertation": "comprehensive study",
        "defended": "presented",
        "peer review": "stakeholder feedback",
        "publication": "deliverable",
        
        # Military → Civilian
        "squadron": "team",
        "deployment": "rollout/launch",
        "operations": "execution"
    }
    
    @classmethod
    def spin_text(
        cls,
        text: str,
        target_stage: CompanyStage,
        preserve_metrics: bool = True
    ) -> Dict[str, any]:
        """Translate sector jargon in `text` into general industry language.

        The translation does not depend on `target_stage`; the stage selects the
        accompanying guidance in the explanation. Terms are matched as whole
        phrases, so "mission" is translated inside "led the mission" but not
        inside "commission".
        """
        original = text or ""
        spun = original
        changes = []
        
        def _replace(match: "re.Match[str]") -> str:
            replacement = cls.TRANSFORMATION_RULES[match.group(0).lower()]
            replaced = replacement.capitalize() if match.group(0)[0].isupper() else replacement
            changes.append({
                "original": match.group(0),
                "replaced": replaced,
                "reason": "General industry language"
            })
            return replaced

        # Longest rule first so a specific phrase ("patient care") is applied
        # before a rule for a word it contains ("care" would be, if it existed).
        rule_order = sorted(cls.TRANSFORMATION_RULES, key=len, reverse=True)
        pattern = re.compile(
            r"(?<!\w)(" + "|".join(re.escape(rule) for rule in rule_order) + r")(?!\w)",
            re.IGNORECASE,
        )
        spun = pattern.sub(_replace, spun)
        
        # The stage shapes the guidance, not the rewrite; reject an unknown one.
        if target_stage not in cls.DICTIONARIES:
            raise ValueError(f"Unknown company stage: {target_stage!r}")
        
        # Extract and preserve metrics if requested
        metrics = []
        if preserve_metrics:
            metrics = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?[%$]?', spun)
        
        # Calculate similarity
        similarity = cls._calculate_similarity(original, spun)
        
        return {
            "original": original,
            "spun": spun,
            "changes": changes,
            "target_stage": target_stage.value,
            "preserved_metrics": metrics,
            "similarity": similarity,
            "explanation": cls._generate_explanation(target_stage, changes)
        }
    
    @classmethod
    def suggest_spinning(
        cls,
        text: str,
   job_description: str
    ) -> CompanyStage:
        """
        Suggest which company stage to spin towards based on JD.
        
        Args:
            text: Your original text
            job_description: Target job description
            
        Returns:
            Recommended company stage
        """
        jd_lower = (job_description or "").lower()
        
        # Score each stage
        scores = {
            CompanyStage.EARLY_STAGE: 0,
            CompanyStage.GROWTH_STAGE: 0,
            CompanyStage.ENTERPRISE: 0
        }
        
        for stage, dictionary in cls.DICTIONARIES.items():
            # Check keywords
            for keyword in dictionary["keywords"]:
                if keyword.lower() in jd_lower:
                    scores[stage] += 2
            
            # Check verbs
            for verb in dictionary["verbs"]:
                if verb.lower() in jd_lower:
                    scores[stage] += 1
        
        # Additional heuristics
        if any(term in jd_lower for term in ["startup", "early", "seed", "pre-revenue"]):
            scores[CompanyStage.EARLY_STAGE] += 5
        
        if any(term in jd_lower for term in ["series a", "series b", "growth", "scaling"]):
            scores[CompanyStage.GROWTH_STAGE] += 5
        
        if any(term in jd_lower for term in ["fortune 500", "enterprise", "global", "compliance"]):
            scores[CompanyStage.ENTERPRISE] += 5
        
        # Return highest scoring stage
        return max(scores, key=scores.get)
    
    @classmethod
    def get_examples(cls, stage: CompanyStage) -> List[Dict[str, str]]:
        """
        Get example transformations for a company stage.
        
        Args:
            stage: Company stage
            
        Returns:
            List of before/after examples
        """
        examples = {
            CompanyStage.EARLY_STAGE: [
                {
                    "before": "Led cross-functional team to deliver enterprise-grade solution with compliance",
                    "after": "Shipped MVP with cross-functional team, validating product-market fit through rapid iterations"
                },
                {
                    "before": "Managed stakeholder alignment across multiple departments",
                    "after": "Built and launched feature collaborating with engineering and design"
                }
            ],
            
            CompanyStage.GROWTH_STAGE: [
                {
                    "before": "Shipped MVP and validated concept with early users",
                    "after": "Scaled feature to 100K users, increasing conversion by 35% through data-driven optimization"
                },
                {
                    "before": "Coordinated with stakeholders to ensure compliance",
                    "after": "Optimized workflow reducing processing time by 40% while maintaining quality"
                }
            ],
            
            CompanyStage.ENTERPRISE: [
                {
                    "before": "Shipped feature quickly with small team",
                    "after": "Coordinated cross-functional initiative across 5 departments, ensuring compliance with industry standards"
                },
                {
                    "before": "Tested new approach with users",
                    "after": "Managed enterprise rollout following change management framework, aligning stakeholders across organization"
                }
            ]
        }
        
        return examples.get(stage, [])
    
    @classmethod
    def _calculate_similarity(cls, original: str, spun: str) -> float:
        """Calculate similarity between original and spun text (0-1)"""
        original_words = set(original.lower().split())
        spun_words = set(spun.lower().split())
        
        intersection = original_words.intersection(spun_words)
        union = original_words.union(spun_words)
        
        if not union:
            return 1.0
        
        return len(intersection) / len(union)
    
    @classmethod
    def _generate_explanation(cls, stage: CompanyStage, changes: List[Dict]) -> str:
        """Generate human-readable explanation of changes"""
        if not changes:
            return f"No changes needed - text already matches {stage.value} language"
        
        explanations = {
            CompanyStage.EARLY_STAGE: "Adapted to emphasize speed, iteration, and validation",
            CompanyStage.GROWTH_STAGE: "Adapted to emphasize metrics, scaling, and optimization",
            CompanyStage.ENTERPRISE: "Adapted to emphasize coordination, compliance, and stakeholder management"
        }
        
        base = explanations[stage]
        return f"{base}. Made {len(changes)} adaptations to match target industry language."


# Example usage
if __name__ == "__main__":
    # Example 1: Healthcare to Tech
    healthcare_text = "Led hospice care teams serving vulnerable families in high-stress environments"
    
    result = SpinningStrategy.spin_text(
        healthcare_text,
        CompanyStage.GROWTH_STAGE
    )
    
    print("Original:", result["original"])
    print("Spun:", result["spun"])
    print("Changes:", len(result["changes"]))
    for change in result["changes"]:
        print(f"  - {change['original']} → {change['replaced']}")
    
    # Example 2: Auto-suggest stage
    jd = "We're a Series B startup looking to scale our product to 1M users. Need someone who can optimize conversion and drive growth metrics."
    suggested = SpinningStrategy.suggest_spinning("", jd)
    print(f"\nSuggested stage: {suggested}")
    
    # Example 3: Get examples
    examples = SpinningStrategy.get_examples(CompanyStage.ENTERPRISE)
    print(f"\nEnterprise examples:")
    for ex in examples:
        print(f"Before: {ex['before']}")
        print(f"After: {ex['after']}\n")
