"""
6-Point Bullet Validation Service

Comprehensive validation for resume bullets using the 6-point framework.
Ensures every bullet meets quality standards before being used.
"""

import re
from typing import Dict, List, Tuple
from schemas import (
    SixPointBullet, 
    BulletValidationResult, 
    MetricsDetectionResult
)


class BulletValidator:
    """Validates bullets against the 6-point framework and quality standards"""
    
    # Character count requirements
    MIN_CHARS = 240
    MAX_CHARS = 260
    IDEAL_CHARS = 250
    
    # Strong action verbs (recommended starters)
    STRONG_VERBS = [
        "led", "built", "designed", "developed", "created", "established",
        "launched", "implemented", "architected", "drove", "spearheaded",
        "orchestrated", "pioneered", "transformed", "optimized", "scaled",
        "increased", "reduced", "improved", "accelerated", "delivered",
        "achieved", "exceeded", "generated", "streamlined", "automated",
        "coordinated", "facilitated", "managed", "directed", "executed"
    ]
    
    # Weak/generic verbs to avoid
    WEAK_VERBS = [
        "helped", "worked on", "responsible for", "assisted with",
        "participated in", "contributed to", "involved in", "dealt with",
        "handled", "did", "made", "got", "had"
    ]
    
    # Generic/weak phrases to avoid
    GENERIC_PHRASES = [
        "various tasks", "day-to-day", "as needed", "duties included",
        "worked closely", "team player", "hard worker", "detail-oriented",
        "self-starter", "go-getter", "think outside the box"
    ]
    
    @classmethod
    def validate_bullet(cls, bullet: SixPointBullet) -> BulletValidationResult:
        """
        Comprehensive validation of a 6-point bullet.
        
        Args:
            bullet: SixPointBullet object to validate
            
        Returns:
            BulletValidationResult with detailed feedback
        """
        errors = []
        warnings = []
        suggestions = []
        auto_fix = {}
        
        # Assemble the full bullet text
        full_text = cls._assemble_bullet(bullet)
        char_count = len(full_text)
        
        # Check 1: All 6 points present
        all_six_points = cls._check_all_six_points(bullet, errors)
        
        # Check 2: Character count
        char_count_valid = cls._check_character_count(char_count, errors, warnings, suggestions)
        
        # Check 3: Metrics presence
        metrics_result = cls._detect_metrics(full_text)
        has_metrics = metrics_result.has_metrics
        if not has_metrics:
            errors.append("Bullet must contain metrics (numbers, percentages, dollar amounts)")
            suggestions.extend(metrics_result.suggestions_if_missing)
        
        # Check 4: Strong action verb
        strong_verb = cls._check_action_verb(bullet.action, warnings, suggestions)
        
        # Check 5: No generic language
        no_generic = cls._check_generic_language(full_text, warnings, suggestions)
        
        # Check 6: Result field has metrics
        result_has_metrics = cls._check_result_has_metrics(bullet.result, errors, suggestions)
        
        # Check 7: Method is descriptive
        cls._check_method_quality(bullet.method, warnings, suggestions)
        
        # Check 8: Impact is meaningful
        cls._check_impact_quality(bullet.impact, warnings, suggestions)
        
        # Calculate quality score
        quality_score = cls._calculate_quality_score(
            char_count=char_count,
            has_metrics=has_metrics,
            has_all_six=all_six_points,
            strong_verb=strong_verb,
            no_generic=no_generic,
            result_has_metrics=result_has_metrics
        )
        
        # Auto-fix suggestions
        auto_fix_available = False
        if char_count > cls.MAX_CHARS:
            auto_fix["character_count"] = f"Trim to {cls.MAX_CHARS} characters"
            auto_fix_available = True
        
        if not strong_verb and bullet.action:
            similar_strong = cls._suggest_strong_verb(bullet.action)
            if similar_strong:
                auto_fix["action_verb"] = f"Replace with: {similar_strong}"
                auto_fix_available = True
        
        # Overall validation
        is_valid = (
            all_six_points and
            char_count_valid and
            has_metrics and
            quality_score >= 70
        )
        
        return BulletValidationResult(
            is_valid=is_valid,
            character_count=char_count,
            has_metrics=has_metrics,
            has_all_six_points=all_six_points,
            has_strong_verb=strong_verb,
            no_generic_language=no_generic,
            quality_score=quality_score,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            auto_fix_available=auto_fix_available,
            auto_fix_suggestions=auto_fix
        )
    
    @classmethod
    def _assemble_bullet(cls, bullet: SixPointBullet) -> str:
        """Assemble the 6 points into a single bullet text"""
        parts = [
            bullet.action,
            bullet.context,
            bullet.method,
            bullet.result,
            bullet.impact,
            bullet.outcome
        ]
        
        # Join with smart punctuation
        text = f"{bullet.action} {bullet.context}"
        
        if bullet.method:
            text += f", {bullet.method}"
        
        if bullet.result:
            text += f", {bullet.result}"
        
        if bullet.impact:
            text += f", {bullet.impact}"
        
        if bullet.outcome:
            text += f" {bullet.outcome}"
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\s+,', ',', text)
        
        return text
    
    @classmethod
    def _check_all_six_points(cls, bullet: SixPointBullet, errors: List[str]) -> bool:
        """Check if all 6 points are present and non-empty"""
        missing = []
        
        if not bullet.action or len(bullet.action.strip()) == 0:
            missing.append("Action")
        if not bullet.context or len(bullet.context.strip()) == 0:
            missing.append("Context")
        if not bullet.method or len(bullet.method.strip()) == 0:
            missing.append("Method")
        if not bullet.result or len(bullet.result.strip()) == 0:
            missing.append("Result")
        if not bullet.impact or len(bullet.impact.strip()) == 0:
            missing.append("Impact")
        if not bullet.outcome or len(bullet.outcome.strip()) == 0:
            missing.append("Outcome")
        
        if missing:
            errors.append(f"Missing required fields: {', '.join(missing)}")
            return False
        
        return True
    
    @classmethod
    def _check_character_count(
        cls, 
        count: int, 
        errors: List[str], 
        warnings: List[str],
        suggestions: List[str]
    ) -> bool:
        """Check if character count is within acceptable range"""
        if count < cls.MIN_CHARS:
            errors.append(f"Bullet too short ({count} chars). Must be at least {cls.MIN_CHARS} characters.")
            suggestions.append("Add more detail to context, method, or impact")
            return False
        
        if count > cls.MAX_CHARS:
            errors.append(f"Bullet too long ({count} chars). Must be under {cls.MAX_CHARS} characters.")
            suggestions.append("Trim less important details or use more concise language")
            return False
        
        if count < cls.MIN_CHARS + 10:
            warnings.append(f"Bullet is close to minimum length ({count} chars)")
        
        if count > cls.MAX_CHARS - 10:
            warnings.append(f"Bullet is close to maximum length ({count} chars)")
        
        return True
    
    @classmethod
    def _detect_metrics(cls, text: str) -> MetricsDetectionResult:
        """Detect metrics (numbers, percentages, dollar amounts) in text"""
        metrics = []
        metric_types = []
        
        # Percentage pattern
        percentage_pattern = r'\d+(?:\.\d+)?%'
        percentages = re.findall(percentage_pattern, text)
        if percentages:
            metrics.extend(percentages)
            metric_types.append("percentage")
        
        # Dollar amount pattern
        dollar_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?[KMB]?'
        dollars = re.findall(dollar_pattern, text)
        if dollars:
            metrics.extend(dollars)
            metric_types.append("dollar")
        
        # Large numbers with commas
        number_pattern = r'\d+(?:,\d{3})+'
        large_numbers = re.findall(number_pattern, text)
        if large_numbers:
            metrics.extend(large_numbers)
            metric_types.append("number")
        
        # Numbers with units (M, K, B for millions, thousands, billions)
        unit_pattern = r'\d+(?:\.\d+)?[KMB](?:\+)?'
        unit_numbers = re.findall(unit_pattern, text)
        if unit_numbers:
            metrics.extend(unit_numbers)
            metric_types.append("scaled_number")
        
        # Ratios (e.g., 3:1, 5x)
        ratio_pattern = r'\d+:\d+|\d+x'
        ratios = re.findall(ratio_pattern, text, re.IGNORECASE)
        if ratios:
            metrics.extend(ratios)
            metric_types.append("ratio")
        
        # Plain numbers (as backup)
        if not metrics:
            plain_numbers = re.findall(r'\b\d+\b', text)
            if plain_numbers:
                metrics.extend(plain_numbers[:3])  # Limit to first 3
                metric_types.append("plain_number")
        
        has_metrics = len(metrics) > 0
        
        suggestions = []
        if not has_metrics:
            suggestions = [
                "Add specific numbers: How many? How much?",
                "Include percentages: By what %?",
                "Quantify impact: How many users, dollars, hours saved?",
                "Example: 'reduced time by 40%', 'grew revenue by $500K', 'served 10K+ users'"
            ]
        
        return MetricsDetectionResult(
            has_metrics=has_metrics,
            metrics_found=metrics,
            metric_types=list(set(metric_types)),
            suggestions_if_missing=suggestions
        )
    
    @classmethod
    def _check_action_verb(
        cls, 
        action: str, 
        warnings: List[str], 
        suggestions: List[str]
    ) -> bool:
        """Check if action starts with a strong verb"""
        if not action:
            return False
        
        action_lower = action.lower().strip()
        first_word = action_lower.split()[0] if action_lower.split() else ""
        
        # Check if starts with weak verb
        if any(weak in action_lower for weak in cls.WEAK_VERBS):
            warnings.append(f"Weak action verb: '{action}'. Use a stronger, more specific verb.")
            suggestions.append(f"Try: {', '.join(cls.STRONG_VERBS[:5])}")
            return False
        
        # Check if starts with strong verb
        if first_word in cls.STRONG_VERBS:
            return True
        
        # Not weak but not in strong list either
        warnings.append(f"Consider using a more impactful action verb")
        return True  # Not invalid, just not optimal
    
    @classmethod
    def _check_generic_language(
        cls, 
        text: str, 
        warnings: List[str], 
        suggestions: List[str]
    ) -> bool:
        """Check for generic/weak phrases"""
        text_lower = text.lower()
        found_generic = []
        
        for phrase in cls.GENERIC_PHRASES:
            if phrase in text_lower:
                found_generic.append(phrase)
        
        if found_generic:
            warnings.append(f"Generic language detected: {', '.join(found_generic)}")
            suggestions.append("Replace generic phrases with specific, quantified achievements")
            return False
        
        return True
    
    @classmethod
    def _check_result_has_metrics(
        cls, 
        result: str, 
        errors: List[str],
        suggestions: List[str]
    ) -> bool:
        """Ensure the result field specifically contains metrics"""
        if not result:
            return False
        
        metrics = cls._detect_metrics(result)
        if not metrics.has_metrics:
            errors.append("Result field must contain specific metrics or numbers")
            suggestions.append("Add quantified outcome: 'reducing X by Y%', 'increasing Z to N'")
            return False
        
        return True
    
    @classmethod
    def _check_method_quality(cls, method: str, warnings: List[str], suggestions: List[str]):
        """Check if method is descriptive enough"""
        if not method:
            return
        
        if len(method) < 10:
            warnings.append("Method field is very brief. Add more detail about how you did it.")
            suggestions.append("Example: 'using Agile methodology', 'through stakeholder interviews', 'by implementing automation'")
    
    @classmethod
    def _check_impact_quality(cls, impact: str, warnings: List[str], suggestions: List[str]):
        """Check if impact is meaningful"""
        if not impact:
            return
        
        if len(impact) < 10:
            warnings.append("Impact field is very brief. Describe the business effect.")
            suggestions.append("Example: 'improving team productivity', 'enhancing customer satisfaction', 'reducing operational costs'")
    
    @classmethod
    def _calculate_quality_score(
        cls,
        char_count: int,
        has_metrics: bool,
        has_all_six: bool,
        strong_verb: bool,
        no_generic: bool,
        result_has_metrics: bool
    ) -> int:
        """Calculate overall quality score (0-100)"""
        score = 0
        
        # All 6 points present (30 points)
        if has_all_six:
            score += 30
        
        # Has metrics (25 points)
        if has_metrics:
            score += 25
        
        # Result specifically has metrics (20 points)
        if result_has_metrics:
            score += 20
        
        # Character count in range (10 points)
        if cls.MIN_CHARS <= char_count <= cls.MAX_CHARS:
            score += 10
            # Bonus for being close to ideal
            if abs(char_count - cls.IDEAL_CHARS) <= 5:
                score += 5
        
        # Strong action verb (5 points)
        if strong_verb:
            score += 5
        
        # No generic language (5 points)
        if no_generic:
            score += 5
        
        return min(100, score)
    
    @classmethod
    def _suggest_strong_verb(cls, current_action: str) -> str:
        """Suggest a strong verb to replace weak one"""
        if not current_action:
            return "Led"
        
        current_lower = current_action.lower()
        
        # Simple mapping of weak to strong
        suggestions_map = {
            "helped": "Enabled",
            "worked on": "Developed",
            "assisted": "Supported",
            "responsible for": "Managed",
            "participated": "Contributed",
            "handled": "Managed",
            "did": "Executed",
            "made": "Created"
        }
        
        for weak, strong in suggestions_map.items():
            if weak in current_lower:
                return strong
        
        # Default suggestions
        return "Led"
    
    @classmethod
    def auto_fix_bullet(cls, bullet: SixPointBullet) -> Tuple[SixPointBullet, List[str]]:
        """
        Attempt to automatically fix common issues.
        
        Returns:
            Tuple of (fixed_bullet, list_of_changes_made)
        """
        changes = []
        fixed = bullet.model_copy(deep=True) if hasattr(bullet, "model_copy") else bullet.copy(deep=True)
        
        # Fix 1: Trim if too long
        full_text = cls._assemble_bullet(fixed)
        if len(full_text) > cls.MAX_CHARS:
            # Try trimming outcome first (least critical)
            if len(fixed.outcome) > 20:
                fixed.outcome = fixed.outcome[:20] + "..."
                changes.append("Trimmed outcome to fit character limit")
            
            # Recalculate
            full_text = cls._assemble_bullet(fixed)
            if len(full_text) > cls.MAX_CHARS:
                # Trim impact
                if len(fixed.impact) > 20:
                    fixed.impact = fixed.impact[:20] + "..."
                    changes.append("Trimmed impact to fit character limit")
        
        # Fix 2: Suggest strong verb
        if fixed.action:
            action_lower = fixed.action.lower().strip()
            first_word = action_lower.split()[0] if action_lower else ""
            if first_word in cls.WEAK_VERBS or any(action_lower.startswith(wv) for wv in cls.WEAK_VERBS):
                suggested = cls._suggest_strong_verb(fixed.action)
                original = fixed.action
                # Don't auto-replace, just suggest
                changes.append(f"Suggestion: Replace '{original}' with '{suggested}'")
        
        return fixed, changes


# Example usage and tests
if __name__ == "__main__":
    # Test bullet
    test_bullet = SixPointBullet(
        action="Led",
        context="cross-functional team for payment reconciliation platform",
        method="using Agile methodology and facilitated stakeholder interviews",
        result="reducing manual processing time by 40%",
        impact="improving cash flow visibility",
        outcome="for Fortune 500 clients"
    )
    
    result = BulletValidator.validate_bullet(test_bullet)
    
    print("Validation Result:")
    print(f"Valid: {result.is_valid}")
    print(f"Quality Score: {result.quality_score}/100")
    print(f"Character Count: {result.character_count}")
    print(f"Has Metrics: {result.has_metrics}")
    print(f"\nErrors: {result.errors}")
    print(f"Warnings: {result.warnings}")
    print(f"Suggestions: {result.suggestions}")
    
    # Test metrics detection
    text_with_metrics = "Increased revenue by 150% ($2.5M) and reduced costs by $500K, serving 10K+ users"
    metrics = BulletValidator._detect_metrics(text_with_metrics)
    print(f"\nMetrics found: {metrics.metrics_found}")
    print(f"Types: {metrics.metric_types}")
