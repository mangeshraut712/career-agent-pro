from fastapi import APIRouter, HTTPException, Body, Depends, status
import logging
from typing import Dict, Any

from services.job_service import JobService
from services.jd_assessor import assess_job_fit
from services.bullet_framework import BulletFramework
from services.competency_assessor import CompetencyAssessor
from dependencies import get_job_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/extract-job")
async def extract_job(
    url_data: dict = Body(...),
    job_service: JobService = Depends(get_job_service)
):
    """Extract job description from URL."""
    try:
        url = url_data.get("url")
        if not url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="URL is required"
            )
        
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid URL format"
            )
        
        result = await job_service.extract_from_url(url)
        logger.info(f"Successfully extracted job from URL: {url}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to extract job: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract job description from URL"
        )

@router.post("/assess-job-fit")
async def assess_job_fit_endpoint(data: dict = Body(...)):
    """
    JD Assessment - Analyze job description and score candidate fit.
    
    Returns:
    - Fit score (0-100)
    - Fit level (excellent/strong/moderate/weak)
    - Competency breakdown with weights
    - Strengths and gaps
    - Spinning recommendation (startup/growth/enterprise language)
    - Action items for improvement
    - Recommended bullet distribution for resume
    """
    try:
        job_description = data.get("job_description", "")
        resume_data = data.get("resume_data", {})
        
        if not job_description:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="job_description is required"
            )
        
        assessment = assess_job_fit(job_description, resume_data)
        logger.info(f"JD assessment completed - Fit score: {assessment['fit_score']}")
        return assessment
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to assess job fit: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assess job fit"
        )

@router.post("/detect-company-stage")
async def detect_company_stage(data: dict = Body(...)):
    """
    Detect company stage from job description for spinning strategy.
    
    Returns: early_stage, growth_stage, or enterprise
    """
    try:
        job_description = data.get("job_description", "")
        
        if not job_description:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="job_description is required"
            )
        
        stage = BulletFramework.detect_company_stage(job_description)
        keywords = BulletFramework.get_spinning_keywords(stage)
        
        return {
            "stage": stage.value,
            "spinning_keywords": keywords,
            "recommendation": f"Use {stage.value.replace('_', ' ')} language in your resume"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to detect company stage: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to detect company stage"
        )

@router.post("/assess-competencies")
async def assess_competencies_endpoint(data: dict = Body(...)):
    """
    Assess JD competencies and calculate fit score.
    
    Returns:
    - List of competencies with weightage (%)
    - Company stage (early/growth/enterprise)
    - Overall fit score (if user data provided)
    - Top strengths and gaps
    - Recommendations
    """
    try:
        jd_text = data.get("job_description", "")
        requirements = data.get("requirements", [])
        skills = data.get("skills", [])
        
        if not jd_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="job_description is required"
            )
        
        # Assess JD
        assessment = CompetencyAssessor.assess_job_description(jd_text, requirements, skills)
        
        # Calculate fit if user data provided
        user_skills = data.get("user_skills", [])
        user_experience = data.get("user_experience", [])
        
        if user_skills or user_experience:
            fit_result = CompetencyAssessor.calculate_fit_score(
                assessment,
                user_skills,
                user_experience
            )
            assessment["fit_analysis"] = fit_result
        
        logger.info(f"Competency assessment completed - Stage: {assessment['company_stage']}")
        
        return assessment
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to assess competencies: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assess competencies"
        )
