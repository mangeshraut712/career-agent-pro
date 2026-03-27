"""Main FastAPI application for CareerAgentPro backend."""
import sys
import os
import logging
from functools import lru_cache

# Add current directory to path for serverless/deployment environments
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Body, File, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
from schemas import EnhancementRequest, CoverLetterRequest, CommunicationRequest, ResumeData, BulletSelectionRequest, SixPointBullet
from services.ai_service import AIService
from services.job_service import JobService
from services.export_service import ExportService
from services.autofill_service import AutofillService
from services.resume_parser import ResumeParser
from services.bullet_framework import (
    BulletFramework, CompanyStage, MetricDiversifier,
    ActionVerbChecker, analyze_complete_resume
)
from services.bullet_library import BulletLibrary
from services.verification_service import ResumeVerifier as LegacyResumeVerifier, CoverLetterVerifier, OutreachVerifier
from services.jd_assessor import JDAssessor, assess_job_fit
from services.outreach_service import OutreachCreator, generate_outreach_strategy
from services.cover_letter_service import CoverLetterService
from services.workflow_orchestrator import ApplicationOrchestrator
# New quality framework services
from services.bullet_validator import BulletValidator
from services.competency_assessor import CompetencyAssessor
from services.spinning_service import SpinningStrategy
from services.resume_verifier import ResumeVerifier as QualityResumeVerifier
from services.bullet_library_manager import BulletLibraryManager
# LangChain service layer
from services.langchain_service import (
    analyze_job_fit as langchain_analyze_job_fit,
    tailor_resume as langchain_tailor_resume,
    generate_cover_letter_langchain,
    run_full_pipeline as langchain_full_pipeline,
)
# Streaming router
from routers.streaming import router as streaming_router

load_dotenv()

# Configure logging with optimized format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CareerAgentPro API",
    version="1.0.0",
    description="AI-Powered Career Platform API",
    # Disable docs in production for faster startup
    docs_url="/docs" if os.getenv("DEBUG") else None,
    redoc_url=None,
)

# Register streaming router (SSE endpoints)
app.include_router(streaming_router)

# Add GZip compression for responses > 500 bytes
app.add_middleware(GZipMiddleware, minimum_size=500)

# Configure CORS with optimized settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,  # Cache preflight for 24 hours
)

# Cached service initialization for better performance
@lru_cache()
def get_ai_service():
    return AIService()

@lru_cache()
def get_export_service():
    return ExportService()

# Initialize services lazily
ai_service = get_ai_service()
job_service = JobService(ai_service)
export_service = get_export_service()

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation error", "details": exc.errors()},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "message": "An unexpected error occurred."},
    )


@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
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

@app.get("/health")
async def health():
    """Comprehensive health check endpoint with API key status."""
    from datetime import datetime
    from env_loader import get_ai_status
    
    is_vercel = os.getenv("VERCEL") == "1" or os.getenv("ENVIRONMENT") == "production"
    ai_status = get_ai_status()
    
    health_status = {
        "status": "healthy",
        "service": "CareerAgentPro",
        "version": "1.0.0",
        "environment": "vercel" if is_vercel else "development",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "api": "ok",
            "ai_service": ai_status["mode"],
            "api_key_present": ai_status["api_key_present"],
            "api_key_valid": ai_status["api_key_valid"],
        },
        "ai_configuration": {
            "enabled": ai_status["ai_enabled"],
            "model": ai_status["model"],
            "mode": ai_status["mode"],
            "api_key_length": ai_status["api_key_length"],
        },
        "features": {
            "resume_parsing": True,
            "resume_enhancement": ai_status["ai_enabled"],
            "job_extraction": True,
            "cover_letter_generation": ai_status["ai_enabled"],
            "export_pdf": True,
            "export_docx": True,
            "export_latex": True,
            "ai_fallbacks": True,  # Always available
        }
    }
    
    if not ai_status["ai_enabled"]:
        health_status["status"] = "operational_with_fallbacks"
        health_status["note"] = "AI features using heuristic fallbacks. Set OPENROUTER_API_KEY for full AI."
        
    return health_status

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to CareerAgentPro API"}

@app.post("/extract-job")
async def extract_job(url_data: dict = Body(...)):
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

@app.post("/enhance-resume")
async def enhance_resume(request: EnhancementRequest):
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

@app.post("/generate-cover-letter")
async def generate_cover_letter(request: CoverLetterRequest):
    """Generate cover letter based on resume and job description."""
    try:
        content = await ai_service.generate_cover_letter(
            request.resume_data,
            request.job_description,
            request.template_type
        )
        logger.info("Cover letter generated successfully")
        return {"content": content}
    except Exception as e:
        logger.error(f"Failed to generate cover letter: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate cover letter"
        )

@app.post("/generate-communication")
async def generate_communication(request: CommunicationRequest):
    """Generate communication (email/LinkedIn message) based on resume and job."""
    try:
        if request.type not in ["email", "linkedin_message", "follow_up"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid communication type. Must be: email, linkedin_message, or follow_up"
            )
        
        content = await ai_service.generate_communication(
            request.resume_data,
            request.job_description,
            request.type
        )
        logger.info(f"Communication ({request.type}) generated successfully")
        return {"content": content}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate communication: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate communication"
        )

@app.post("/export/docx")
async def export_docx(resume: ResumeData):
    """Export resume as DOCX file."""
    try:
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            output_path = tmp_file.name
        
        export_service.to_docx(resume, output_path)
        
        response = FileResponse(
            output_path,
            filename=f"{resume.name.replace(' ', '_')}_resume.docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        logger.info(f"DOCX export completed for: {resume.name}")
        return response
    except Exception as e:
        logger.error(f"Failed to export DOCX: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export resume as DOCX"
        )

@app.post("/export/pdf")
async def export_pdf(resume: ResumeData):
    """Export resume as PDF file."""
    try:
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            output_path = tmp_file.name
        
        export_service.to_pdf(resume, output_path)
        
        response = FileResponse(
            output_path,
            filename=f"{resume.name.replace(' ', '_')}_resume.pdf",
            media_type="application/pdf"
        )
        logger.info(f"PDF export completed for: {resume.name}")
        return response
    except Exception as e:
        logger.error(f"Failed to export PDF: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export resume as PDF"
        )

@app.post("/export/latex")
async def export_latex(resume: ResumeData):
    """Export resume as LaTeX file."""
    try:
        content = export_service.to_latex(resume)
        logger.info(f"LaTeX export completed for: {resume.name}")
        return PlainTextResponse(content, media_type="text/plain")
    except Exception as e:
        logger.error(f"Failed to export LaTeX: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export resume as LaTeX"
        )

@app.post("/generate-autofill")
async def generate_autofill(data: dict = Body(...)):
    """Generate autofill script for job application platforms."""
    try:
        resume_data = data.get("resume_data")
        platform = data.get("platform", "google")
        
        if not resume_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="resume_data is required"
            )
        
        if platform not in ["google", "indeed", "linkedin"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid platform. Supported: google, indeed, linkedin"
            )
        
        script = AutofillService.generate_autofill_script(resume_data, platform)
        logger.info(f"Autofill script generated for platform: {platform}")
        return {"script": script}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate autofill script: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate autofill script"
        )


# ============================================================================
# NEW APPLY-PILOT INSPIRED ENDPOINTS
# ============================================================================

@app.post("/assess-job-fit")
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


@app.post("/analyze-bullets")
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


@app.post("/detect-company-stage")
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


@app.post("/analyze-complete-resume")
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

@app.post("/bullet-library/select-for-job")
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


@app.post("/verify-resume")
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


@app.post("/verify-cover-letter")
async def verify_cover_letter(data: dict = Body(...)):
    """
    Cover Letter Verification Gate.
    
    Validates:
    - Line count (8-12 lines)
    - Word count (150-200 words)
    - Personalization (company/role mentions)
    - Proper format (salutation/closing)
    """
    try:
        cover_letter = data.get("cover_letter", "")
        job_data = data.get("job_data")
        
        if not cover_letter:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="cover_letter is required"
            )
        
        report = CoverLetterVerifier.verify(cover_letter, job_data)
        
        return {
            "status": report.overall_status.value,
            "score": report.score,
            "checks": [
                {"name": c.name, "status": c.status.value, "message": c.message}
                for c in report.checks
            ],
            "suggestions": report.suggestions
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to verify cover letter: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify cover letter"
        )


@app.post("/generate-outreach-strategy")
async def generate_outreach_strategy_endpoint(data: dict = Body(...)):
    """
    Multi-Track Outreach Strategy with 3-Tier Escalation.
    
    Returns:
    - Primary track recommendation (direct/warm_intro/cold)
    - Messages for each track and tier
    - Target contacts to find
    - Preparation checklist
    """
    try:
        job_data = data.get("job_data", {})
        resume_data = data.get("resume_data", {})
        
        if not job_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="job_data is required"
            )
        
        strategy = generate_outreach_strategy(job_data, resume_data)
        logger.info(f"Outreach strategy generated for {strategy['company']}")
        return strategy
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate outreach strategy: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate outreach strategy"
        )


@app.post("/verify-outreach")
async def verify_outreach(data: dict = Body(...)):
    """
    Outreach Message Verification Gate.
    
    Validates message length, personalization, and call-to-action.
    """
    try:
        message = data.get("message", "")
        message_type = data.get("type", "linkedin")
        job_data = data.get("job_data")
        
        if not message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="message is required"
            )
        
        report = OutreachVerifier.verify(message, message_type, job_data)
        
        return {
            "status": report.overall_status.value,
            "score": report.score,
            "checks": [
                {"name": c.name, "status": c.status.value, "message": c.message}
                for c in report.checks
            ],
            "suggestions": report.suggestions
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to verify outreach: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify outreach"
        )


@app.post("/orchestrate-application")
async def orchestrate_application(data: dict = Body(...)):
    """
    Autonomous Application Orchestrator.
    Runs JD assessment, bullet selection, CL generation, and outreach strategy.
    """
    try:
        job_description = data.get("job_description")
        resume_data = data.get("resume_data")
        bullet_library = data.get("bullet_library", [])
        
        if not job_description or not resume_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="job_description and resume_data are required"
            )
        
        results = await ApplicationOrchestrator.run_full_pipeline(
            job_description, resume_data, bullet_library
        )
        
        return results
    except Exception as e:
        logger.error(f"Failed to orchestrate application: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to orchestrate application"
        )


@app.post("/generate-elite-cover-letter")
async def generate_elite_cover_letter(data: dict = Body(...)):
    """
    Generates a high-converting cover letter using the 4-paragraph framework.
    """
    try:
        job_data = data.get("job_data")
        resume_data = data.get("resume_data")
        template_type = data.get("template_type", "minimalist")
        company_hook = data.get("company_hook")
        
        if not job_data or not resume_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="job_data and resume_data are required"
            )
            
        result = await CoverLetterService.generate(
            job_data, resume_data, template_type, company_hook
        )
        
        return result
    except Exception as e:
        logger.error(f"Failed to generate elite cover letter: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate cover letter"
        )


# ============================================================================
# NEW QUALITY FRAMEWORK ENDPOINTS
# ============================================================================

@app.post("/validate-bullet")
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
            detail=f"Failed to validate bullet: {str(e)}"
        )


@app.post("/assess-competencies")
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
            detail=f"Failed to assess competencies: {str(e)}"
        )


@app.post("/spin-text")
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
            detail=f"Failed to spin text: {str(e)}"
        )


@app.post("/verify-resume-quality")
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
            detail=f"Failed to verify resume quality: {str(e)}"
        )


# ============================================================================
# LANGCHAIN-POWERED ENDPOINTS (2026 Portfolio Upgrade)
# ============================================================================

@app.post("/langchain/analyze-fit")
async def langchain_analyze_fit_endpoint(data: dict = Body(...)):
    """
    Job fit analysis via LangChain chains.
    Uses structured output parsing for reliable JSON responses.
    """
    try:
        job_description = data.get("job_description", "")
        resume_data = data.get("resume_data", {})

        if not job_description:
            raise HTTPException(status_code=400, detail="job_description is required")

        result = await langchain_analyze_job_fit(job_description, resume_data)
        logger.info("LangChain job fit analysis completed")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LangChain analyze-fit failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/langchain/tailor-resume")
async def langchain_tailor_resume_endpoint(data: dict = Body(...)):
    """
    Resume tailoring via LangChain chains.
    Structured output: summary, skills, experience suggestions, keywords.
    """
    try:
        job_description = data.get("job_description", "")
        resume_data = data.get("resume_data", {})

        if not job_description:
            raise HTTPException(status_code=400, detail="job_description is required")

        result = await langchain_tailor_resume(job_description, resume_data)
        logger.info("LangChain resume tailoring completed")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LangChain tailor-resume failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/langchain/cover-letter")
async def langchain_cover_letter_endpoint(data: dict = Body(...)):
    """
    Cover letter generation via LangChain chains.
    """
    try:
        job_description = data.get("job_description", "")
        resume_data = data.get("resume_data", {})
        template_type = data.get("template_type", "professional")

        if not job_description:
            raise HTTPException(status_code=400, detail="job_description is required")

        result = await generate_cover_letter_langchain(
            job_description, resume_data, template_type
        )
        logger.info("LangChain cover letter generated")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LangChain cover-letter failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/langchain/full-pipeline")
async def langchain_full_pipeline_endpoint(data: dict = Body(...)):
    """
    Full sequential pipeline: fit analysis → resume tailoring → cover letter.
    Demonstrates LangChain chain composition for complex workflows.
    """
    try:
        job_description = data.get("job_description", "")
        resume_data = data.get("resume_data", {})
        template_type = data.get("template_type", "professional")

        if not job_description:
            raise HTTPException(status_code=400, detail="job_description is required")

        result = await langchain_full_pipeline(
            job_description, resume_data, template_type
        )
        logger.info("LangChain full pipeline completed")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LangChain pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
