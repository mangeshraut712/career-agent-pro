from fastapi import APIRouter, File, UploadFile, HTTPException, Body, Depends, status
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
import logging

from schemas import ResumeData, EnhancementRequest, BulletSelectionRequest, SixPointBullet
from services.ai_service import AIService
from services.resume_parser import ResumeParser
from services.bullet_framework import BulletFramework, analyze_complete_resume, CompanyStage
from services.bullet_library import BulletLibrary
from services.verification_service import ResumeVerifier as LegacyResumeVerifier
from services.bullet_validator import BulletValidator
from services.spinning_service import SpinningStrategy
from services.resume_verifier import ResumeVerifier as QualityResumeVerifier
from dependencies import get_ai_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/parse-resume-stream")
async def parse_resume_stream(
    file: UploadFile = File(...),
    ai_service: AIService = Depends(get_ai_service)
):
    """Stream resume parsing progress and results via SSE."""
    try:
        content = await file.read()
        filename = file.filename or "resume.pdf"
        text = ResumeParser.extract_text(content, filename)
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text")

        return StreamingResponse(
            ai_service.stream_parse_resume(text),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Streaming parse error: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse resume")

@router.post("/parse-resume")
async def parse_resume(
    file: UploadFile = File(...),
    ai_service: AIService = Depends(get_ai_service)
):
    """Parse resume file and extract structured data."""
    try:
        # Read file content
        content = await file.read()
        filename = file.filename or "unknown.pdf"
        
        # Extract text from file
        try:
            text = ResumeParser.extract_text(content, filename)
        except Exception as extract_err:
            raise HTTPException(
                status_code=400, 
                detail=f"Could not read file. Supported formats: PDF, DOCX, TXT. Error: {type(extract_err).__name__}"
            )
        
        if not text or not text.strip():
            raise HTTPException(
                status_code=400, 
                detail="Could not extract text from resume. Please ensure the file contains readable text."
            )
        
        # Parse the extracted text
        try:
            parsed_data = await ai_service.parse_resume(text)
            logger.info(f"Successfully parsed resume: {filename}")
            return parsed_data
        except Exception as parse_err:
            logger.error(f"Error parsing resume: {str(parse_err)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Error parsing resume content: {type(parse_err).__name__}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing resume: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error processing resume: {type(e).__name__}"
        )

@router.post("/enhance-resume")
async def enhance_resume(
    request: EnhancementRequest,
    ai_service: AIService = Depends(get_ai_service)
):
    """Enhance resume based on job description."""
    try:
        result = await ai_service.enhance_resume(request.resume_data, request.job_description)
        logger.info("Resume enhancement completed successfully")
        return result
    except Exception as e:
        logger.error(f"Failed to enhance resume: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enhance resume"
        )

@router.post("/analyze-bullets")
async def analyze_bullets(data: dict = Body(...)):
    """
    6-Point Bullet Framework Analysis.
    
    Analyzes resume bullets against the framework:
    - Action verb
    - Context (scope/environment)
    - Method (approach used)
    - Result (quantifiable outcome)
    - Impact (who/what affected)
    - Business Outcome (strategic value)
    
    Character limit: 240-260 characters per bullet.
    """
    try:
        bullets = data.get("bullets", [])
        
        if not bullets:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="bullets array is required"
            )
        
        if isinstance(bullets, str):
            bullets = [b.strip() for b in bullets.split("\n") if b.strip()]
        
        # Analyze each bullet
        analyses = []
        for bullet in bullets:
            analysis = BulletFramework.analyze_bullet(bullet)
            analyses.append({
                "original": analysis.original,
                "character_count": analysis.character_count,
                "score": analysis.score,
                "framework_checks": {
                    "action": analysis.has_action,
                    "context": analysis.has_context,
                    "method": analysis.has_method,
                    "result": analysis.has_result,
                    "impact": analysis.has_impact,
                    "business_outcome": analysis.has_business_outcome
                },
                "has_metric": analysis.has_metric,
                "suggestions": analysis.suggestions
            })
        
        # Aggregate stats
        batch_stats = BulletFramework.validate_bullet_batch(bullets)
        
        logger.info(f"Analyzed {len(bullets)} bullets - Avg score: {batch_stats['average_score']}")
        return {
            "analyses": analyses,
            "summary": batch_stats
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze bullets: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze bullets"
        )

@router.post("/analyze-complete-resume")
async def analyze_complete_resume_endpoint(data: dict = Body(...)):
    """
    Complete Resume Analysis - ALL Quality Checks.
    
    Combines:
    - 6-Point Bullet Framework
    - Metric Diversity (5 types: TIME, VOLUME, FREQUENCY, SCOPE, QUALITY)
    - Action Verb Uniqueness (no repeats across 13 bullets)
    
    Returns overall score and submission-readiness.
    """
    try:
        bullets = data.get("bullets", [])
        
        if not bullets:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="bullets array is required"
            )
        
        if isinstance(bullets, str):
            bullets = [b.strip() for b in bullets.split("\n") if b.strip()]
        
        # Run complete analysis
        analysis = analyze_complete_resume(bullets)
        
        logger.info(
            f"Complete analysis: Overall={analysis['overall_score']}, "
            f"Ready={analysis['ready_for_submission']}"
        )
        
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze complete resume: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze complete resume"
        )

@router.post("/bullet-library/select-for-job")
async def select_bullets_for_job(request: BulletSelectionRequest):
    """Select best-fit bullets for a job description."""
    try:
        if not request.job_description:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="job_description is required"
            )

        bullets_payload = [b.model_dump() for b in request.bullets]
        result = BulletLibrary.select_for_job(
            bullets=bullets_payload,
            job_description=request.job_description,
            count=request.count or 13
        )

        logger.info(
            "Bullet selection completed - selected %s bullets",
            result.get("total_selected", 0)
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to select bullets: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to select bullets for job"
        )

@router.post("/verify-resume")
async def verify_resume(data: dict = Body(...)):
    """
    Resume Verification Gate - Quality checks before submission.
    
    Validates:
    - Required sections present
    - Contact information complete
    - Bullet counts (2-5 per role)
    - Character counts (240-260 per bullet)
    - Skills listed
    - ATS compatibility
    """
    try:
        resume_data = data.get("resume_data", {})
        
        if not resume_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="resume_data is required"
            )
        
        report = LegacyResumeVerifier.verify(resume_data)
        
        return {
            "status": report.overall_status.value,
            "score": report.score,
            "auto_retry": report.auto_retry_recommended,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details
                }
                for c in report.checks
            ],
            "suggestions": report.suggestions
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to verify resume: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify resume"
        )

@router.post("/validate-bullet")
async def validate_bullet_endpoint(data: dict = Body(...)):
    """
    Validate a 6-point bullet against quality standards.
    
    Checks:
    - All 6 points present
    - Character count (240-260)
    - Metrics requirement
    - Strong action verb
    - No generic language
    
    Returns quality score and detailed feedback.
    """
    try:
        bullet_data = data.get("bullet")
        if not bullet_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="bullet object is required"
            )
        
        # Create SixPointBullet object
        bullet = SixPointBullet(**bullet_data)
        
        # Validate
        validation = BulletValidator.validate_bullet(bullet)
        
        logger.info(f"Bullet validated - Quality: {validation.quality_score}/100")
        
        return {
            "is_valid": validation.is_valid,
            "character_count": validation.character_count,
            "has_metrics": validation.has_metrics,
            "has_all_six_points": validation.has_all_six_points,
            "has_strong_verb": validation.has_strong_verb,
            "no_generic_language": validation.no_generic_language,
            "quality_score": validation.quality_score,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "suggestions": validation.suggestions,
            "auto_fix_available": validation.auto_fix_available,
            "auto_fix_suggestions": validation.auto_fix_suggestions
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate bullet: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate bullet"
        )

@router.post("/spin-text")
async def spin_text_endpoint(data: dict = Body(...)):
    """
    Adapt text to match target company stage language.
    
    Transforms language without changing facts:
    - Early Stage: Speed, iteration, validation
    - Growth Stage: Metrics, scaling, optimization
    - Enterprise: Coordination, compliance, stakeholder management
    
    Returns before/after comparison with explanations.
    """
    try:
        text = data.get("text", "")
        target_stage = data.get("targetStage", "growth_stage")
        
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="text is required"
            )
        
        # Map stage string to enum
        stage_map = {
            "early_stage": CompanyStage.EARLY_STAGE,
            "growth_stage": CompanyStage.GROWTH_STAGE,
            "enterprise": CompanyStage.ENTERPRISE
        }
        
        stage_enum = stage_map.get(target_stage, CompanyStage.GROWTH_STAGE)
        
        # Spin the text
        result = SpinningStrategy.spin_text(text, stage_enum)
        
        logger.info(f"Text spun to {target_stage} - {len(result['changes'])} changes made")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to spin text: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to spin text"
        )

@router.post("/verify-resume-quality")
async def verify_resume_quality_endpoint(data: dict = Body(...)):
    """
    Comprehensive resume quality verification.
    
    Validates:
    - Profile completeness
    - All bullets meet standards
    - Overall quality score (0-100)
    - Export readiness
    
    Returns detailed quality report with suggestions.
    """
    try:
        resume_data_dict = data.get("resume")
        bullets_data = data.get("bullets", [])
        strict_mode = data.get("strict_mode", False)
        
        if not resume_data_dict:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="resume object is required"
            )
        
        # Create ResumeData object
        resume = ResumeData(**resume_data_dict)
        
        # Create SixPointBullet objects if provided
        bullets = None
        if bullets_data:
            bullets = [SixPointBullet(**b) for b in bullets_data]
        
        # Verify
        result = QualityResumeVerifier.verify_resume(resume, bullets, strict_mode)
        
        logger.info(
            f"Resume verified - Score: {result['overall_quality_score']}/100, "
            f"Can export: {result['can_export']}"
        )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to verify resume quality: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify resume quality"
        )
