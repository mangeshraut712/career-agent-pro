"""
Bullet Library Service - Selects the best bullets for a job description.
Inspired by Apply-Pilot's bullet library workflow.
"""

import re
from typing import Dict, Any, List, Tuple

from services.jd_assessor import JDAssessor, term_in_text
from services.bullet_framework import BulletFramework


class BulletLibrary:
    """Bullet library selection utilities."""

    @classmethod
    def select_for_job(
        cls,
        bullets: List[Dict[str, Any]],
        job_description: str,
        count: int = 13
    ) -> Dict[str, Any]:
        """Select best-fit bullets based on JD competency weights."""
        cleaned_bullets = cls._normalize_bullets(bullets)
        if not cleaned_bullets:
            return {
                "selected_bullets": [],
                "distribution": {},
                "total_requested": count,
                "total_selected": 0,
                "recommendations": ["Add bullets to your library before selecting for a job."]
            }

        weights = cls._compute_competency_weights(job_description)
        distribution = cls._calculate_distribution(weights, count)
        scored = cls._score_bullets(cleaned_bullets, weights)

        selected = cls._select_by_distribution(scored, distribution, count)

        recommendations = []
        if len(selected) < count:
            recommendations.append(
                f"Only {len(selected)} bullets available. Add more bullets to reach {count}."
            )

        return {
            "selected_bullets": selected,
            "distribution": distribution,
            "total_requested": count,
            "total_selected": len(selected),
            "recommendations": recommendations
        }

    @classmethod
    def _normalize_bullets(cls, bullets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize bullet inputs."""
        normalized = []
        for bullet in bullets:
            text = (bullet.get("text") or "").strip()
            if not text:
                continue
            tags = bullet.get("tags") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            competency = bullet.get("competency")
            normalized.append({
                "id": bullet.get("id") or text[:50],
                "text": text,
                "tags": [t.lower() for t in tags],
                "competency": competency
            })
        return normalized

    @classmethod
    def _compute_competency_weights(cls, job_description: str) -> Dict[str, float]:
        """Compute JD-driven weights for each competency area."""
        jd_lower = (job_description or "").lower()
        weights = {}

        for area, config in JDAssessor.COMPETENCY_AREAS.items():
            keywords = config.get("keywords", [])
            mentions = sum(1 for kw in keywords if term_in_text(jd_lower, kw))
            base_weight = config.get("weight", 0.2)
            if mentions > 0:
                weights[area] = base_weight * (mentions / max(len(keywords), 1))
            else:
                weights[area] = 0.0

        if sum(weights.values()) == 0:
            weights = {
                area: config.get("weight", 0.2)
                for area, config in JDAssessor.COMPETENCY_AREAS.items()
            }

        return weights

    @classmethod
    def _calculate_distribution(cls, weights: Dict[str, float], total: int) -> Dict[str, int]:
        """Allocate bullet counts by competency weights."""
        weighted = {k: v for k, v in weights.items() if v > 0}
        if not weighted:
            return {"general": total}

        total_weight = sum(weighted.values())
        distribution = {}
        for area, weight in weighted.items():
            allocation = int((weight / total_weight) * total)
            if allocation > 0:
                distribution[area] = allocation

        remaining = total - sum(distribution.values())
        if remaining > 0 and distribution:
            top_area = max(distribution.keys(), key=lambda k: distribution[k])
            distribution[top_area] += remaining

        return distribution

    @classmethod
    def _score_bullets(
        cls,
        bullets: List[Dict[str, Any]],
        weights: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Score bullets against each competency area."""
        scored = []
        for bullet in bullets:
            analysis = BulletFramework.analyze_bullet(bullet["text"])
            area_scores = {}
            area_matches = {}

            for area, weight in weights.items():
                keywords = JDAssessor.COMPETENCY_AREAS.get(area, {}).get("keywords", [])
                keyword_score, matched = cls._keyword_score(bullet["text"], bullet["tags"], keywords)

                competency_boost = 10 if bullet.get("competency") == area else 0
                combined_score = (keyword_score * 0.6) + (analysis.score * 0.4) + competency_boost
                area_scores[area] = round(min(100, combined_score), 1)
                area_matches[area] = matched

            best_area, best_score = cls._best_area(area_scores)

            scored.append({
                "id": bullet["id"],
                "text": bullet["text"],
                "tags": bullet["tags"],
                "competency": bullet.get("competency") or best_area,
                "analysis_score": analysis.score,
                "area_scores": area_scores,
                "area_matches": area_matches,
                "best_area": best_area,
                "best_score": best_score
            })

        return scored

    @classmethod
    def _keyword_score(
        cls,
        text: str,
        tags: List[str],
        keywords: List[str]
    ) -> Tuple[float, List[str]]:
        """Score bullet against a keyword set."""
        text_lower = text.lower()
        matched = []

        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
            if re.search(pattern, text_lower):
                matched.append(keyword)

        tag_hits = [tag for tag in tags if tag in [k.lower() for k in keywords]]
        raw_score = len(matched) + (0.5 * len(tag_hits))

        if not keywords:
            return 0.0, matched

        score = (raw_score / len(keywords)) * 100
        return round(min(100, score), 1), matched

    @classmethod
    def _best_area(cls, area_scores: Dict[str, float]) -> Tuple[str, float]:
        if not area_scores:
            return "general", 0.0
        best_area = max(area_scores.keys(), key=lambda k: area_scores[k])
        return best_area, area_scores[best_area]

    @classmethod
    def _select_by_distribution(
        cls,
        scored: List[Dict[str, Any]],
        distribution: Dict[str, int],
        total: int
    ) -> List[Dict[str, Any]]:
        """Select bullets based on distribution and best scores."""
        selected = []
        selected_ids = set()

        for area, target in distribution.items():
            if area == "general":
                candidates = sorted(scored, key=lambda b: b["best_score"], reverse=True)
            else:
                candidates = sorted(
                    scored,
                    key=lambda b: b["area_scores"].get(area, 0),
                    reverse=True
                )
            picked = 0
            for bullet in candidates:
                if bullet["id"] in selected_ids:
                    continue
                selected.append({
                    "id": bullet["id"],
                    "text": bullet["text"],
                    "competency": area,
                    "score": bullet["area_scores"].get(area, 0),
                    "analysis_score": bullet["analysis_score"],
                    "matched_keywords": bullet.get("area_matches", {}).get(area, [])
                })
                selected_ids.add(bullet["id"])
                picked += 1
                if picked >= target:
                    break

        if len(selected) < total:
            remaining = sorted(
                [b for b in scored if b["id"] not in selected_ids],
                key=lambda b: b["best_score"],
                reverse=True
            )
            for bullet in remaining:
                if len(selected) >= total:
                    break
                selected.append({
                    "id": bullet["id"],
                    "text": bullet["text"],
                    "competency": bullet["best_area"],
                    "score": bullet["best_score"],
                    "analysis_score": bullet["analysis_score"],
                    "matched_keywords": bullet.get("area_matches", {}).get(bullet["best_area"], [])
                })
                selected_ids.add(bullet["id"])

        return selected
