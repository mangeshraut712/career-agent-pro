"""
Cover Letter Service - 4-Paragraph Framework (Hook, Value, Alignment, CTA).
Based on Aakash Gupta's methodology used in Apply-Pilot.
"""

import logging
from typing import Dict, Any, List, Optional
from services.ai_service import AIService

logger = logging.getLogger(__name__)

class CoverLetterService:
    """Creates company-specific cover letters using the 4-paragraph framework."""
    
    TEMPLATES = {
        "minimalist": {
            "name": "Minimalist Standard",
            "description": "Clean, concise, 4-paragraph prose format.",
            "structure": "Hook -> Value -> Alignment -> CTA"
        },
        "enterprise": {
            "name": "B2B/Enterprise Focus",
            "description": "Focuses on strategic enterprise growth and scalability.",
            "structure": "Hook -> Value -> Alignment -> CTA"
        },
        "technical": {
            "name": "Technical/Platform Company",
            "description": "Emphasizes technical PM achievements and engineering collaboration.",
            "structure": "Hook -> Value -> Alignment -> CTA"
        }
    }

    @classmethod
    async def generate(
        cls,
        job_data: Optional[Dict[str, Any]] = None,
        resume_data: Optional[Dict[str, Any]] = None,
        template_type: str = "minimalist",
        company_hook: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a cover letter using the 4-paragraph framework."""
        safe_job = job_data or {}
        safe_resume = resume_data or {}
        
        prompt = cls._build_prompt(safe_job, safe_resume, template_type, company_hook)
        
        try:
            ai_service = AIService()
            content = await ai_service.generate_text(prompt)
            
            # Clean up the output (remove AI artifacts)
            content = cls._clean_content(content)
            
            return {
                "content": content,
                "template": template_type if template_type in cls.TEMPLATES else "minimalist",
                "structure": "Hook -> Value -> Alignment -> CTA",
                "rules": [
                    "8-12 lines total",
                    "150-200 words",
                    "No formal headers or H2 titles"
                ]
            }
        except Exception as e:
            logger.error(f"Failed to generate cover letter: {str(e)}")
            raise

    @classmethod
    def _build_prompt(
        cls,
        job_data: Optional[Dict[str, Any]],
        resume_data: Optional[Dict[str, Any]],
        template_type: str,
        company_hook: Optional[str]
    ) -> str:
        """Build the AI prompt for cover letter generation."""
        template = cls.TEMPLATES.get(template_type, cls.TEMPLATES["minimalist"])
        safe_job = job_data or {}
        safe_resume = resume_data or {}

        raw_desc = safe_job.get('description') or ''
        trimmed_desc = str(raw_desc)[:500]

        raw_exp = safe_resume.get('experience')
        exp_list = raw_exp if isinstance(raw_exp, list) else []
        top_achievements = str(exp_list[:2])
        
        return f"""
        You are an expert Career Agent. Create a high-converting cover letter based on Aakash Gupta's methodology.
        
        STRUCTURE:
        Paragraph 1: COMPANY-SPECIFIC HOOK (2-3 sentences)
        Paragraph 2: VALUE PROPOSITION (2-3 sentences - strongest relevant achievement + quantified impact)
        Paragraph 3: COMPANY ALIGNMENT (2-3 sentences - why this company + how you address their specific challenge)
        Paragraph 4: CALL TO ACTION (1-2 sentences - professional but confident)
        
        STRICT CONSTRAINTS:
        - 8-12 lines total.
        - 150-200 words.
        - NO formal headers (No "Dear...", No "Re:", No "Subject:").
        - NO H2 section titles.
        - Minimalist prose format only.
        - Tone: Professional, personable, and confident.
        
        CONTEXT:
        Job Title: {safe_job.get('title', 'Senior Product Manager')}
        Company: {safe_job.get('company', 'Target Company')}
        Job Requirements: {trimmed_desc}
        Company Research/Hook: {company_hook if company_hook else 'General company growth and mission'}
        
        CANDIDATE DATA:
        Name: {safe_resume.get('name', 'Candidate')}
        Top Achievements: {top_achievements}
        
        TEMPLATE STYLE: {template['name']} ({template['description']})
        
        OUTPUT ONLY THE COVER LETTER TEXT.
        """

    @classmethod
    def _clean_content(cls, content: Optional[str]) -> str:
        """Remove AI prefixes or suffixes."""
        if not content:
            return ""
        # Remove common AI prefixes
        prefixes = ["Here is a cover letter", "Sure, here is", "Cover letter:", "Subject:"]
        lines = content.strip().split('\n')
        while lines and any(lines[0].strip().startswith(p) for p in prefixes):
            lines.pop(0)
            
        # Remove markdown markers if any
        content = "\n".join(lines).strip()
        content = content.replace("```markdown", "").replace("```", "").strip()
        
        return content
