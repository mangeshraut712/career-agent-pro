"""
6-Point Bullet Framework Service for Resume Optimization.

Inspired by Apply-Pilot's proven methodology for creating impactful resume bullets.
Each bullet must include: Action + Context + Method + Result + Impact + Business Outcome

Character limit: 240-260 characters per bullet for optimal ATS compatibility.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CompanyStage(Enum):
    """Company stage for spinning strategy."""
    EARLY_STAGE = "early_stage"  # Startup bullets
    GROWTH_STAGE = "growth_stage"  # Scaling/metrics bullets
    ENTERPRISE = "enterprise"  # Fortune 500 bullets


@dataclass
class BulletAnalysis:
    """Analysis result for a resume bullet."""
    original: str
    character_count: int
    has_action: bool
    has_context: bool
    has_method: bool
    has_result: bool
    has_impact: bool
    has_business_outcome: bool
    has_metric: bool
    score: float  # 0-100
    suggestions: List[str]
    enhanced: Optional[str] = None


@dataclass
class CompetencyArea:
    """Competency area with weightage."""
    name: str
    weight: float  # 0.0-1.0
    keywords: List[str]
    matched_bullets: List[str] = None

    def __post_init__(self):
        if self.matched_bullets is None:
            self.matched_bullets = []


class BulletFramework:
    """
    6-Point Bullet Framework for creating impactful resume bullets.
    
    Framework:
    1. Action - Strong action verb (Led, Developed, Implemented)
    2. Context - Scope and environment (cross-functional, enterprise-wide)
    3. Method - Specific approach used (using Agile, leveraging AI)
    4. Result - Quantifiable outcome (reduced by 40%, increased by $1M)
    5. Impact - Who/what was affected (for 500+ users, across 10 teams)
    6. Business Outcome - Strategic value (improving efficiency, reducing costs)
    """
    
    # Character limits
    MIN_CHARS = 240
    MAX_CHARS = 260
    OPTIMAL_CHARS = 250
    
    # Action verbs by category
    ACTION_VERBS = {
        "leadership": [
            "Led", "Directed", "Championed", "Spearheaded", "Orchestrated",
            "Mentored", "Guided", "Supervised", "Managed", "Oversaw"
        ],
        "achievement": [
            "Achieved", "Attained", "Exceeded", "Surpassed", "Delivered",
            "Earned", "Won", "Secured", "Accomplished", "Realized"
        ],
        "creation": [
            "Developed", "Created", "Designed", "Built", "Established",
            "Initiated", "Launched", "Pioneered", "Introduced", "Invented"
        ],
        "improvement": [
            "Optimized", "Enhanced", "Improved", "Streamlined", "Transformed",
            "Revamped", "Modernized", "Upgraded", "Refined", "Elevated"
        ],
        "analysis": [
            "Analyzed", "Evaluated", "Assessed", "Researched", "Investigated",
            "Examined", "Identified", "Discovered", "Diagnosed", "Mapped"
        ],
        "communication": [
            "Presented", "Negotiated", "Collaborated", "Coordinated", "Facilitated",
            "Advocated", "Influenced", "Persuaded", "Communicated", "Articulated"
        ],
        "technical": [
            "Implemented", "Engineered", "Automated", "Deployed", "Integrated",
            "Architected", "Configured", "Programmed", "Coded", "Debugged"
        ]
    }
    
    # Context indicators
    CONTEXT_PATTERNS = [
        r"cross-functional",
        r"enterprise-wide",
        r"global",
        r"company-wide",
        r"department",
        r"team of \d+",
        r"\d+ stakeholder",
        r"Fortune \d+",
        r"startup",
        r"scale-up"
    ]
    
    # Method indicators
    METHOD_PATTERNS = [
        r"using",
        r"leveraging",
        r"applying",
        r"through",
        r"via",
        r"with",
        r"by implementing",
        r"utilizing",
        r"employing",
        r"adopting"
    ]
    
    # Metric patterns (Result)
    METRIC_PATTERNS = [
        r"\d+%",
        r"\$[\d,]+[KMB]?",
        r"[\d,]+\s*(users|customers|clients)",
        r"\d+x",
        r"reduced by \d+",
        r"increased by \d+",
        r"saved \$?[\d,]+",
        r"\d+\s*(hours|days|weeks|months)",
        r"from \d+ to \d+"
    ]
    
    # Impact indicators
    IMPACT_PATTERNS = [
        r"for \d+",
        r"across \d+ (teams|departments|regions)",
        r"serving \d+",
        r"impacting \d+",
        r"enabling \d+",
        r"supporting \d+"
    ]
    
    # Business outcome indicators
    BUSINESS_OUTCOME_PATTERNS = [
        r"improving",
        r"reducing costs",
        r"increasing revenue",
        r"enhancing efficiency",
        r"accelerating",
        r"driving growth",
        r"enabling scale",
        r"ensuring compliance",
        r"mitigating risk",
        r"competitive advantage"
    ]
    
    # Spinning keywords by company stage
    SPINNING_KEYWORDS = {
        CompanyStage.EARLY_STAGE: [
            "agile", "scrappy", "MVP", "iterate", "pivot", "bootstrap",
            "lean", "rapid", "prototype", "validate", "hustle", "wear many hats"
        ],
        CompanyStage.GROWTH_STAGE: [
            "scaled", "10x", "100x", "growth", "expansion", "hypergrowth",
            "Series", "OKRs", "KPIs", "metrics-driven", "data-driven"
        ],
        CompanyStage.ENTERPRISE: [
            "Fortune 500", "enterprise-wide", "global", "compliance",
            "governance", "stakeholder", "cross-functional", "executive"
        ]
    }

    @classmethod
    def analyze_bullet(cls, bullet: str) -> BulletAnalysis:
        """
        Analyze a resume bullet against the 6-point framework.
        
        Returns detailed analysis with scores and suggestions.
        """
        bullet = bullet.strip()
        char_count = len(bullet)
        
        # Check each framework point
        has_action = cls._check_action(bullet)
        has_context = cls._check_context(bullet)
        has_method = cls._check_method(bullet)
        has_result = cls._check_result(bullet)
        has_impact = cls._check_impact(bullet)
        has_business_outcome = cls._check_business_outcome(bullet)
        has_metric = cls._check_metric(bullet)
        
        # Calculate score
        points = [has_action, has_context, has_method, has_result, has_impact, has_business_outcome]
        base_score = (sum(points) / 6) * 70  # 70% for framework compliance
        
        # Character count bonus/penalty
        char_score = 0
        if cls.MIN_CHARS <= char_count <= cls.MAX_CHARS:
            char_score = 20  # Perfect range
        elif char_count < cls.MIN_CHARS:
            char_score = max(0, 20 - (cls.MIN_CHARS - char_count) * 0.5)
        else:
            char_score = max(0, 20 - (char_count - cls.MAX_CHARS) * 0.5)
        
        # Metric bonus
        metric_score = 10 if has_metric else 0
        
        total_score = min(100, base_score + char_score + metric_score)
        
        # Generate suggestions
        suggestions = cls._generate_suggestions(
            bullet, has_action, has_context, has_method,
            has_result, has_impact, has_business_outcome,
            has_metric, char_count
        )
        
        return BulletAnalysis(
            original=bullet,
            character_count=char_count,
            has_action=has_action,
            has_context=has_context,
            has_method=has_method,
            has_result=has_result,
            has_impact=has_impact,
            has_business_outcome=has_business_outcome,
            has_metric=has_metric,
            score=round(total_score, 1),
            suggestions=suggestions
        )
    
    @classmethod
    def _check_action(cls, bullet: str) -> bool:
        """Check if bullet starts with a strong action verb."""
        first_word = bullet.split()[0] if bullet.split() else ""
        first_word = first_word.strip(",.:;")
        
        all_verbs = []
        for verbs in cls.ACTION_VERBS.values():
            all_verbs.extend(verbs)
        
        return first_word in all_verbs
    
    @classmethod
    def _check_context(cls, bullet: str) -> bool:
        """Check if bullet includes context."""
        bullet_lower = bullet.lower()
        return any(re.search(pattern, bullet_lower) for pattern in cls.CONTEXT_PATTERNS)
    
    @classmethod
    def _check_method(cls, bullet: str) -> bool:
        """Check if bullet describes the method used."""
        bullet_lower = bullet.lower()
        return any(re.search(pattern, bullet_lower) for pattern in cls.METHOD_PATTERNS)
    
    @classmethod
    def _check_result(cls, bullet: str) -> bool:
        """Check if bullet includes quantifiable results."""
        return any(re.search(pattern, bullet) for pattern in cls.METRIC_PATTERNS)
    
    @classmethod
    def _check_impact(cls, bullet: str) -> bool:
        """Check if bullet describes impact scope."""
        bullet_lower = bullet.lower()
        return any(re.search(pattern, bullet_lower) for pattern in cls.IMPACT_PATTERNS)
    
    @classmethod
    def _check_business_outcome(cls, bullet: str) -> bool:
        """Check if bullet includes business outcome."""
        bullet_lower = bullet.lower()
        return any(re.search(pattern, bullet_lower) for pattern in cls.BUSINESS_OUTCOME_PATTERNS)
    
    @classmethod
    def _check_metric(cls, bullet: str) -> bool:
        """Check if bullet contains quantifiable metrics."""
        return any(re.search(pattern, bullet) for pattern in cls.METRIC_PATTERNS)
    
    @classmethod
    def _generate_suggestions(
        cls, bullet: str, has_action: bool, has_context: bool,
        has_method: bool, has_result: bool, has_impact: bool,
        has_business_outcome: bool, has_metric: bool, char_count: int
    ) -> List[str]:
        """Generate actionable improvement suggestions."""
        suggestions = []
        
        if not has_action:
            suggestions.append(
                "Start with a strong action verb like 'Led', 'Developed', 'Optimized', or 'Implemented'"
            )
        
        if not has_context:
            suggestions.append(
                "Add context about scope (e.g., 'cross-functional team', 'enterprise-wide', 'for Fortune 500 client')"
            )
        
        if not has_method:
            suggestions.append(
                "Specify the method/approach (e.g., 'using Agile methodology', 'leveraging data analytics')"
            )
        
        if not has_result or not has_metric:
            suggestions.append(
                "Add quantifiable results (e.g., 'reducing costs by 40%', 'increasing efficiency by 2x')"
            )
        
        if not has_impact:
            suggestions.append(
                "Clarify the impact scope (e.g., 'for 500+ users', 'across 10 departments')"
            )
        
        if not has_business_outcome:
            suggestions.append(
                "Connect to business value (e.g., 'improving customer satisfaction', 'driving revenue growth')"
            )
        
        if char_count < cls.MIN_CHARS:
            suggestions.append(
                f"Expand bullet to {cls.MIN_CHARS}-{cls.MAX_CHARS} characters (currently {char_count})"
            )
        elif char_count > cls.MAX_CHARS:
            suggestions.append(
                f"Condense bullet to {cls.MIN_CHARS}-{cls.MAX_CHARS} characters (currently {char_count})"
            )
        
        return suggestions
    
    @classmethod
    def get_spinning_keywords(cls, stage: CompanyStage) -> List[str]:
        """Get spinning keywords for a specific company stage."""
        return cls.SPINNING_KEYWORDS.get(stage, [])
    
    @classmethod
    def detect_company_stage(cls, job_description: str) -> CompanyStage:
        """Detect company stage from job description for spinning strategy."""
        jd_lower = job_description.lower()
        
        scores = {
            CompanyStage.EARLY_STAGE: 0,
            CompanyStage.GROWTH_STAGE: 0,
            CompanyStage.ENTERPRISE: 0
        }
        
        for stage, keywords in cls.SPINNING_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in jd_lower:
                    scores[stage] += 1
        
        # Additional heuristics
        if any(term in jd_lower for term in ["seed", "pre-seed", "angel", "small team"]):
            scores[CompanyStage.EARLY_STAGE] += 3
        
        if any(term in jd_lower for term in ["series a", "series b", "series c", "hypergrowth"]):
            scores[CompanyStage.GROWTH_STAGE] += 3
        
        if any(term in jd_lower for term in ["fortune", "global", "multinational", "corporate"]):
            scores[CompanyStage.ENTERPRISE] += 3
        
        # Return stage with highest score
        return max(scores, key=scores.get)
    
    @classmethod
    def calculate_competency_weights(
        cls, job_description: str, competency_areas: List[Dict[str, Any]]
    ) -> List[CompetencyArea]:
        """
        Analyze job description and assign weights to competency areas.
        
        Returns weighted competencies for bullet selection.
        """
        jd_lower = job_description.lower()
        total_mentions = 0
        
        competencies = []
        for area in competency_areas:
            competency = CompetencyArea(
                name=area["name"],
                weight=0.0,
                keywords=area.get("keywords", [])
            )
            
            # Count keyword mentions
            mentions = 0
            for keyword in competency.keywords:
                mentions += jd_lower.count(keyword.lower())
            
            competency.weight = mentions
            total_mentions += mentions
            competencies.append(competency)
        
        # Normalize weights
        if total_mentions > 0:
            for competency in competencies:
                competency.weight = round(competency.weight / total_mentions, 2)
        else:
            # Equal distribution if no keywords found
            equal_weight = 1.0 / len(competencies) if competencies else 0
            for competency in competencies:
                competency.weight = equal_weight
        
        # Sort by weight descending
        competencies.sort(key=lambda c: c.weight, reverse=True)
        
        return competencies
    
    @classmethod
    def validate_bullet_batch(cls, bullets: List[str]) -> Dict[str, Any]:
        """
        Validate a batch of bullets and return aggregate statistics.
        """
        analyses = [cls.analyze_bullet(b) for b in bullets]
        
        total_score = sum(a.score for a in analyses) / len(analyses) if analyses else 0

        def share(flag: str) -> float:
            # An empty batch has no compliant bullets; avoid dividing by zero.
            if not analyses:
                return 0
            return sum(1 for a in analyses if getattr(a, flag)) / len(analyses) * 100

        framework_compliance = {
            "action": share("has_action"),
            "context": share("has_context"),
            "method": share("has_method"),
            "result": share("has_result"),
            "impact": share("has_impact"),
            "business_outcome": share("has_business_outcome"),
        }
        
        char_distribution = {
            "under_min": sum(1 for a in analyses if a.character_count < cls.MIN_CHARS),
            "optimal": sum(1 for a in analyses if cls.MIN_CHARS <= a.character_count <= cls.MAX_CHARS),
            "over_max": sum(1 for a in analyses if a.character_count > cls.MAX_CHARS)
        }
        
        return {
            "total_bullets": len(bullets),
            "average_score": round(total_score, 1),
            "framework_compliance": framework_compliance,
            "character_distribution": char_distribution,
            "with_metrics": sum(1 for a in analyses if a.has_metric),
            "individual_analyses": [
                {
                    "bullet": a.original[:50] + "..." if len(a.original) > 50 else a.original,
                    "score": a.score,
                    "char_count": a.character_count,
                    "suggestions_count": len(a.suggestions)
                }
                for a in analyses
            ]
        }


# Example bullet templates
EXAMPLE_BULLETS = {
    "leadership": """Led cross-functional discovery for payment reconciliation platform, facilitating 15+ stakeholder interviews using Jobs-to-be-Done framework to identify friction points, reducing manual processing time by 40% and improving cash flow visibility for Fortune 500 clients.""",
    
    "technical": """Architected and deployed microservices infrastructure using Kubernetes and AWS ECS, implementing zero-downtime deployment pipelines that reduced release cycles from 2 weeks to 2 days, enabling 95% faster feature delivery across 12 product teams.""",
    
    "growth": """Spearheaded customer acquisition strategy leveraging data-driven persona targeting and A/B testing, increasing trial-to-paid conversion by 35% and reducing CAC by $120 per customer, driving $2.4M in additional ARR within 6 months.""",
    
    "operations": """Streamlined global supply chain operations by implementing predictive analytics for demand forecasting, reducing inventory holding costs by $1.2M annually while maintaining 99.5% order fulfillment rate across 8 distribution centers."""
}


class MetricDiversifier:
    """
    Ensure metric diversity across resume bullets.
    Based on Apply-Pilot's 5 metric types framework.
    """
    
    METRIC_TYPES = {
        'time': r'\d+\s*(?:min|mins|minute|hour|hrs|day|days|week|weeks|month|months|year|years|hr|ms|sec)',
        'volume': r'\d+[KMB]?\+?\s*(?:users|customers|transactions|sessions|pageviews|records|requests|queries|applications|submissions)',
        'frequency': r'\d+\+?\s*(?:per|weekly|monthly|daily|times|×|x)',
        'scope': r'\d+\+?\s*(?:markets|countries|teams|states|cities|departments|industries|segments|regions)',
        'quality': r'\d+%\s*(?:up from|down from|increase|decrease|improvement|retention|accuracy|satisfaction|precision|recall|conversion)'
    }
    
    @classmethod
    def classify_metric(cls, bullet: str) -> Optional[str]:
        """Classify the primary metric type in a bullet."""
        for metric_type, pattern in cls.METRIC_TYPES.items():
            if re.search(pattern, bullet, re.IGNORECASE):
                return metric_type
        return None
    
    @classmethod
    def check_diversity(cls, bullets: List[str]) -> Dict[str, Any]:
        """Check metric diversity across all bullets."""
        from collections import defaultdict
        
        metric_distribution = defaultdict(int)
        classified_bullets = []
        
        for i, bullet in enumerate(bullets):
            metric_type = cls.classify_metric(bullet)
            if metric_type:
                metric_distribution[metric_type] += 1
                classified_bullets.append({
                    'bullet_index': i + 1,
                    'metric_type': metric_type,
                    'snippet': bullet[:60] + '...' if len(bullet) > 60 else bullet
                })
        
        warnings = []
        recommendations = []
        
        for metric_type, count in metric_distribution.items():
            if count > 2:
                warnings.append(f"⚠️ {metric_type.title()} metrics appear {count} times (recommended max: 2)")
                recommendations.append(f"Consider varying some {metric_type} metrics to other types")
        
        missing_types = set(cls.METRIC_TYPES.keys()) - set(metric_distribution.keys())
        if missing_types:
            recommendations.append(f"Consider adding {', '.join(missing_types)} metrics for better diversity")
        
        diversity_score = (len(metric_distribution) / 5) * 100
        
        return {
            'distribution': dict(metric_distribution),
            'diversity_score': round(diversity_score, 1),
            'total_bullets': len(bullets),
            'classified_bullets': classified_bullets,
            'warnings': warnings,
            'recommendations': recommendations,
            'all_types_present': len(missing_types) == 0
        }


class ActionVerbChecker:
    """Ensure action verb uniqueness across bullets."""
    
    STRONG_ACTION_VERBS = [
        'Led', 'Built', 'Designed', 'Spearheaded', 'Drove', 'Launched',
        'Scaled', 'Optimized', 'Implemented', 'Developed', 'Created',
        'Established', 'Orchestrated', 'Pioneered', 'Architected', 'Engineered',
        'Streamlined', 'Executed', 'Delivered', 'Managed', 'Directed',
        'Facilitated', 'Coordinated', 'Analyzed', 'Synthesized', 'Evaluated',
        'Transformed', 'Modernized', 'Automated', 'Integrated', 'Deployed',
        'Championed', 'Initiated', 'Accelerated', 'Enhanced', 'Refined'
    ]
    
    @classmethod
    def extract_action_verb(cls, bullet: str) -> Optional[str]:
        """Extract the action verb (first word) from a bullet."""
        cleaned = bullet.strip().lstrip('•').lstrip('-').lstrip('*').strip()
        words = cleaned.split()
        if words:
            return words[0].strip('.,;:')
        return None
    
    @classmethod
    def check_uniqueness(cls, bullets: List[str]) -> Dict[str, Any]:
        """Check action verb uniqueness across all bullets."""
        verb_usage = {}
        duplicates = []
        unique_count = 0
        
        for i, bullet in enumerate(bullets):
            verb = cls.extract_action_verb(bullet)
            if verb:
                if verb in verb_usage:
                    duplicates.append({
                        'verb': verb,
                        'first_use': verb_usage[verb],
                        'duplicate_at': i + 1,
                        'suggestion': cls._get_alternative_verb(verb)
                    })
                else:
                    verb_usage[verb] = i + 1
                    unique_count += 1
        
        all_unique = len(duplicates) == 0
        suggestions = []
        if duplicates:
            suggestions.append(f"Found {len(duplicates)} duplicate verb(s)")
            for dup in duplicates:
                suggestions.append(f"Bullet {dup['duplicate_at']}: Replace '{dup['verb']}' with '{dup['suggestion']}'")
        
        return {
            'total_bullets': len(bullets),
            'unique_verbs': unique_count,
            'all_unique': all_unique,
            'duplicates': duplicates,
            'suggestions': suggestions,
            'verb_usage': verb_usage
        }
    
    @classmethod
    def _get_alternative_verb(cls, original_verb: str) -> str:
        """Get an alternative action verb."""
        alternatives = {
            'Led': ['Spearheaded', 'Directed', 'Orchestrated'],
            'Built': ['Developed', 'Created', 'Engineered'],
            'Designed': ['Architected', 'Crafted', 'Engineered'],
            'Managed': ['Directed', 'Oversaw', 'Coordinated'],
            'Developed': ['Built', 'Created', 'Engineered'],
            'Implemented': ['Deployed', 'Executed', 'Launched'],
            'Created': ['Built', 'Developed', 'Established'],
            'Improved': ['Enhanced', 'Optimized', 'Refined'],
            'Increased': ['Boosted', 'Accelerated', 'Amplified'],
            'Reduced': ['Decreased', 'Minimized', 'Streamlined']
        }
        
        if original_verb in alternatives:
            return alternatives[original_verb][0]
        
        import random
        return random.choice([v for v in cls.STRONG_ACTION_VERBS if v != original_verb])


def analyze_complete_resume(bullets: List[str]) -> Dict[str, Any]:
    """Complete resume analysis combining all frameworks."""
    batch_analysis = BulletFramework.validate_bullet_batch(bullets)
    metric_diversity = MetricDiversifier.check_diversity(bullets)
    verb_uniqueness = ActionVerbChecker.check_uniqueness(bullets)
    
    framework_score = batch_analysis['average_score']
    diversity_score = metric_diversity['diversity_score']
    uniqueness_score = 100 if verb_uniqueness['all_unique'] else (
        (verb_uniqueness['unique_verbs'] / verb_uniqueness['total_bullets']) * 100
    )
    
    overall_score = (framework_score * 0.5) + (diversity_score * 0.25) + (uniqueness_score * 0.25)
    
    all_recommendations = []
    all_recommendations.extend(batch_analysis.get('recommendations', []))
    all_recommendations.extend(metric_diversity.get('recommendations', []))
    all_recommendations.extend(verb_uniqueness.get('suggestions', []))
    
    return {
        'overall_score': round(overall_score, 1),
        'bullet_framework': batch_analysis,
        'metric_diversity': metric_diversity,
        'verb_uniqueness': verb_uniqueness,
        'all_recommendations': all_recommendations,
        'ready_for_submission': (
            framework_score >= 75 and
            diversity_score >= 60 and
            verb_uniqueness['all_unique']
        )
    }
