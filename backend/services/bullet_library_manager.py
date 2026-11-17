"""
Enhanced Bullet Library Manager

Manages reusable 6-point bullets with smart selection, storage, and CRUD operations.
Integrates with the 6-point framework for perfect quality control.
"""

from typing import Dict, List, Optional
from datetime import datetime
import uuid
from schemas import (
    SixPointBullet,
    BulletLibraryItem,
    BulletSelectionCriteria,
    BulletValidationResult
)
from services.bullet_validator import BulletValidator
from services.competency_assessor import CompetencyAssessor


class BulletLibraryManager:
    """Enhanced bullet library with 6-point framework integration"""
    
    # In-memory storage (replace with database in production)
    _storage: Dict[str, BulletLibraryItem] = {}

    @classmethod
    def clear_library(cls) -> None:
        """Clear all bullets from in-memory storage (useful for testing or reset)"""
        cls._storage.clear()
    
    @classmethod
    def add_bullet(
        cls,
        bullet: SixPointBullet,
        validate: bool = True
    ) -> Dict:
        """
        Add a bullet to the library.
        
        Args:
            bullet: 6-point bullet to add
            validate: Whether to validate before adding
            
        Returns:
            Dict with added bullet and validation result
        """
        result = {
            "success": False,
            "bullet_id": None,
            "validation": None,
            "message": ""
        }

        if bullet is None:
            result["message"] = "Bullet cannot be None"
            return result
        
        validation = None
        # Validate if requested
        if validate:
            validation = BulletValidator.validate_bullet(bullet)
            result["validation"] = validation
            
            if not validation.is_valid:
                result["message"] = f"Bullet failed validation: {', '.join(validation.errors)}"
                return result
        
        # Generate ID
        bullet_id = bullet.id or str(uuid.uuid4())
        
        # Create library item
        library_item = BulletLibraryItem(
            id=bullet_id,
            bullet=bullet,
            created_at=datetime.now().isoformat(),
            usage_count=0,
            quality_score=validation.quality_score if (validate and validation) else 0
        )
        
        # Store
        cls._storage[bullet_id] = library_item
        
        result["success"] = True
        result["bullet_id"] = bullet_id
        result["message"] = "Bullet added successfully"
        
        return result
    
    @classmethod
    def get_bullet(cls, bullet_id: str) -> Optional[BulletLibraryItem]:
        """Get a bullet by ID"""
        return cls._storage.get(bullet_id)
    
    @classmethod
    def update_bullet(
        cls,
        bullet_id: str,
        bullet: SixPointBullet,
        validate: bool = True
    ) -> Dict:
        """Update an existing bullet"""
        if bullet is None:
            return {
                "success": False,
                "message": "Bullet cannot be None"
            }

        if bullet_id not in cls._storage:
            return {
                "success": False,
                "message": f"Bullet {bullet_id} not found"
            }
        
        validation = None
        # Validate if requested
        if validate:
            validation = BulletValidator.validate_bullet(bullet)
            if not validation.is_valid:
                return {
                    "success": False,
                    "message": f"Bullet failed validation: {', '.join(validation.errors)}",
                    "validation": validation
                }
        
        # Update the bullet
        existing = cls._storage[bullet_id]
        existing.bullet = bullet
        existing.quality_score = validation.quality_score if (validate and validation) else existing.quality_score
        
        return {
            "success": True,
            "message": "Bullet updated successfully",
            "validation": validation if validate else None
        }
    
    @classmethod
    def delete_bullet(cls, bullet_id: str) -> Dict:
        """Delete a bullet from library"""
        if bullet_id not in cls._storage:
            return {
                "success": False,
                "message": f"Bullet {bullet_id} not found"
            }
        
        del cls._storage[bullet_id]
        
        return {
            "success": True,
            "message": "Bullet deleted successfully"
        }
    
    @classmethod
    def list_bullets(
        cls,
        competency: Optional[str] = None,
        company_stage: Optional[str] = None,
        min_quality: int = 0,
        tags: List[str] = None
    ) -> List[BulletLibraryItem]:
        """
        List bullets with optional filtering.
        
        Args:
            competency: Filter by competency
            company_stage: Filter by company stage
            min_quality: Minimum quality score
            tags: Filter by tags
            
        Returns:
            List of matching bullets
        """
        bullets = list(cls._storage.values())
        
        # Apply filters
        if competency:
            bullets = [b for b in bullets if b.bullet.competency == competency]
        
        if company_stage:
            bullets = [b for b in bullets if b.bullet.company_stage == company_stage]
        
        if min_quality > 0:
            bullets = [b for b in bullets if b.quality_score >= min_quality]
        
        if tags:
            bullets = [
                b for b in bullets
                if any(tag in b.bullet.tags for tag in tags)
            ]
        
        # Sort by quality score
        bullets.sort(key=lambda b: b.quality_score, reverse=True)
        
        return bullets
    
    @classmethod
    def smart_select(
        cls,
        criteria: BulletSelectionCriteria
    ) -> Dict:
        """
        Smart selection of bullets based on JD and criteria.
        
        Args:
            criteria: Selection criteria including JD
            
        Returns:
            Dict with selected bullets and metadata
        """
        # Assess the JD
        jd_assessment = CompetencyAssessor.assess_job_description(
            jd_text=criteria.job_description,
            skills=criteria.target_competencies
        )
        
        # Get all bullets
        all_bullets = cls.list_bullets(
            company_stage=criteria.company_stage,
            min_quality=criteria.min_quality_score
        )
        
        if not all_bullets:
            return {
                "selected_bullets": [],
                "total_selected": 0,
                "jd_assessment": jd_assessment,
                "message": "No bullets in library. Add bullets first."
            }
        
        # Score bullets against JD competencies
        scored_bullets = []
        for lib_item in all_bullets:
            bullet = lib_item.bullet
            full_text = BulletValidator._assemble_bullet(bullet)
            
            # Calculate relevance score
            relevance = cls._calculate_relevance(
                bullet_text=full_text,
                competency=bullet.competency,
                jd_competencies=jd_assessment.get("competencies", []) if isinstance(jd_assessment, dict) else [],
                target_stage=criteria.company_stage
            )
            
            # Apply usage penalty if preferring unused bullets
            usage_penalty = lib_item.usage_count * 2 if criteria.prefer_unused else 0
            final_score = relevance - usage_penalty
            
            scored_bullets.append({
                "bullet_item": lib_item,
                "relevance_score": relevance,
                "final_score": final_score,
                "usage_count": lib_item.usage_count
            })
        
        # Sort by final score
        scored_bullets.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Select top bullets
        selected_count = min(criteria.count, len(scored_bullets))
        selected = scored_bullets[:selected_count]
        
        # Apply distribution if provided
        if criteria.distribution:
            selected = cls._apply_distribution(
                scored_bullets=scored_bullets,
                distribution=criteria.distribution,
                jd_competencies=jd_assessment.get("competencies", []) if isinstance(jd_assessment, dict) else []
            )
        
        # Mark bullets as used
        for item in selected:
            bullet_id = item["bullet_item"].id
            if bullet_id in cls._storage:
                cls._storage[bullet_id].usage_count += 1
                cls._storage[bullet_id].last_used = datetime.now().isoformat()
        
        return {
            "selected_bullets": [
                {
                    "id": item["bullet_item"].id,
                    "bullet": item["bullet_item"].bullet,
                    "relevance_score": item["relevance_score"],
                    "quality_score": item["bullet_item"].quality_score,
                    "usage_count": item["usage_count"]
                }
                for item in selected
            ],
            "total_selected": len(selected),
            "total_available": len(all_bullets),
            "jd_assessment": jd_assessment,
            "message": f"Selected {len(selected)} bullets based on JD fit"
        }
    
    @classmethod
    def _calculate_relevance(
        cls,
        bullet_text: str,
        competency: str,
        jd_competencies: List[Dict],
        target_stage: Optional[str]
    ) -> float:
        """Calculate how relevant a bullet is to the JD"""
        score = 0.0
        
        bullet_lower = (bullet_text or "").lower()
        
        # Score based on competency match
        for jd_comp in (jd_competencies or []):
            if not isinstance(jd_comp, dict):
                continue
            if jd_comp.get("name") == competency:
                # High relevance if competency matches
                score += 50
                
                # Additional score for keyword matches
                for keyword in jd_comp.get("keywords_found", []):
                    if keyword and str(keyword).lower() in bullet_lower:
                        score += 5
                
                break
        
        # Stage boost
        if target_stage:
            from services.spinning_service import SpinningStrategy
            stage_dict = SpinningStrategy.DICTIONARIES.get(target_stage, {})
            
            # Check for stage-appropriate verbs and keywords
            for verb in stage_dict.get("verbs", []):
                if verb and str(verb).lower() in bullet_lower:
                    score += 3
            
            for keyword in stage_dict.get("keywords", []):
                if keyword and str(keyword).lower() in bullet_lower:
                    score += 2
        
        return min(100.0, score)
    
    @classmethod
    def _apply_distribution(
        cls,
        scored_bullets: List[Dict],
        distribution: List[int],
        jd_competencies: List[Dict]
    ) -> List[Dict]:
        """Apply distribution across roles/sections"""
        selected = []
        bullets_by_comp = {}
        
        # Group bullets by competency
        for item in scored_bullets:
            comp = item["bullet_item"].bullet.competency or "general"
            if comp not in bullets_by_comp:
                bullets_by_comp[comp] = []
            bullets_by_comp[comp].append(item)
        
        # Distribute according to pattern (e.g., [3, 3, 3, 2, 2])
        for count in distribution:
            # Pick from top competency if available
            if jd_competencies and isinstance(jd_competencies[0], dict):
                top_comp = jd_competencies[0].get("name", "general")
                if top_comp in bullets_by_comp and bullets_by_comp[top_comp]:
                    selected.extend(bullets_by_comp[top_comp][:count])
                    bullets_by_comp[top_comp] = bullets_by_comp[top_comp][count:]
                    continue
            
            # Fallback to best available
            for comp_list in bullets_by_comp.values():
                if comp_list:
                    selected.extend(comp_list[:count])
                    comp_list[:] = comp_list[count:]
                    break
        
        return selected
    
    @classmethod
    def get_statistics(cls) -> Dict:
        """Get library statistics"""
        bullets = list(cls._storage.values())
        
        if not bullets:
            return {
                "total_bullets": 0,
                "avg_quality": 0,
                "competency_distribution": {},
                "stage_distribution": {},
                "total_usage": 0
            }
        
        competency_dist = {}
        stage_dist = {}
        
        for item in bullets:
            comp = item.bullet.competency or "uncategorized"
            competency_dist[comp] = competency_dist.get(comp, 0) + 1
            
            stage = item.bullet.company_stage or "any"
            stage_dist[stage] = stage_dist.get(stage, 0) + 1
        
        return {
            "total_bullets": len(bullets),
            "avg_quality": sum(b.quality_score for b in bullets) / len(bullets),
            "competency_distribution": competency_dist,
            "stage_distribution": stage_dist,
            "total_usage": sum(b.usage_count for b in bullets),
            "most_used": max(bullets, key=lambda b: b.usage_count) if bullets else None
        }


# Example usage
if __name__ == "__main__":
    # Add sample bullets
    bullet1 = SixPointBullet(
        action="Led",
        context="cross-functional team for payment reconciliation platform",
        method="using Agile methodology and stakeholder interviews",
        result="reducing processing time by 40%",
        impact="improving cash flow visibility",
        outcome="for Fortune 500 clients",
        competency="Product Strategy",
        company_stage="growth_stage"
    )
    
    result = BulletLibraryManager.add_bullet(bullet1)
    print("Added bullet:", result["message"])
    print("Quality score:", result["validation"].quality_score)
    
    # List bullets
    bullets = BulletLibraryManager.list_bullets()
    print(f"\nTotal bullets in library: {len(bullets)}")
    
    # Smart selection
    criteria = BulletSelectionCriteria(
        job_description="Looking for PM with strong product strategy and data analytics skills",
        count=3,
        company_stage="growth_stage"
    )
    
    selection = BulletLibraryManager.smart_select(criteria)
    print(f"\nSelected {selection['total_selected']} bullets")
    
    # Statistics
    stats = BulletLibraryManager.get_statistics()
    print(f"\nLibrary stats:")
    print(f"  Total: {stats['total_bullets']}")
    print(f"  Avg Quality: {stats['avg_quality']:.1f}")
