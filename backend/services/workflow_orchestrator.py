"""
Application Orchestrator - Autonomous Workflow Engine.
Coordinates JD assessment, resume tailoring, cover letter generation, and outreach strategy.
Inspired by Apply-Pilot's application orchestrator agent.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from services.jd_assessor import JDAssessor
from services.bullet_library import BulletLibrary
from services.verification_service import ResumeVerifier, CoverLetterVerifier, OutreachVerifier, VerificationStatus
from services.cover_letter_service import CoverLetterService
from services.outreach_service import OutreachCreator
try:
    from services.export_service import ExportService
except ImportError:
    ExportService = None

logger = logging.getLogger(__name__)

class ApplicationOrchestrator:
    """Orchestrates the end-to-end job application workflow."""
    
    @classmethod
    async def run_full_pipeline(
        cls,
        job_description: str,
        resume_data: Dict[str, Any],
        bullet_library: List[Dict[str, Any]],
        options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Runs the complete application pipeline autonomously."""
        options = options or {}
        resume_data = dict(resume_data or {})
        bullet_library = list(bullet_library or [])
        job_description = str(job_description or "")
        results = {
            "timestamp": datetime.now().isoformat(),
            "status": "in_progress",
            "steps": []
        }
        
        try:
            # Step 1: JD Assessment
            cls._log_step(results, "JD Assessment", "Analyzing job description and scoring fit...")
            assessment = JDAssessor.assess(job_description, resume_data)
            results["assessment"] = assessment
            
            # Step 2: Bullet Selection
            cls._log_step(results, "Bullet Selection", "Selecting best-fit bullets from library...")
            selection = BulletLibrary.select_for_job(bullet_library, job_description)
            results["selected_bullets"] = selection["selected_bullets"]
            
            # Step 3: Resume Tailoring
            cls._log_step(results, "Resume Tailoring", "Updating resume structure and content...")
            # For now, we simulate tailoring by combining selection with profile
            tailored_resume = resume_data.copy()
            # logic to update resume_data with selected_bullets would go here
            results["tailored_resume"] = tailored_resume
            
            # Step 4: Resume Verification
            cls._log_step(results, "Resume Verification", "Checking resume against quality gates...")
            resume_report = ResumeVerifier.verify(tailored_resume)
            results["resume_verification"] = resume_report
            
            # Step 5: Cover Letter Generation
            cls._log_step(results, "Cover Letter", "Generating company-specific cover letter...")
            cl_result = await CoverLetterService.generate(
                {"title": assessment.fit_level.value, "company": "Target Company", "description": job_description},
                tailored_resume
            )
            results["cover_letter"] = cl_result
            
            # Step 6: Cover Letter Verification
            cls._log_step(results, "CL Verification", "Checking cover letter quality...")
            cl_report = CoverLetterVerifier.verify(cl_result["content"])
            results["cl_verification"] = cl_report
            
            # Step 7: Outreach Strategy
            cls._log_step(results, "Outreach Strategy", "Generating multi-track outreach plan...")
            strategy = OutreachCreator.create_strategy(
                {"title": "Role", "company": "Company"},
                tailored_resume
            )
            results["outreach"] = strategy
            
            results["status"] = "completed"
            return results
            
        except Exception as e:
            logger.error(f"Orchestration failed: {str(e)}")
            results["status"] = "failed"
            results["error"] = str(e)
            return results

    @staticmethod
    def _log_step(results: Dict, name: str, status: str):
        results["steps"].append({
            "name": name,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
