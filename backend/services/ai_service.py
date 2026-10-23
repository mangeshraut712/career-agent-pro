import os
import json
import re
import asyncio
import logging
from openai import AsyncOpenAI
from typing import Dict, Any, List, Optional
from schemas import ResumeData, JobDescription

# PERMANENT SOLUTION: Import from bulletproof env loader
from env_loader import get_api_key, is_ai_enabled, get_ai_status

logger = logging.getLogger(__name__)


class AIService:
    """AI Service using OpenRouter API for resume parsing and enhancement.
    
    PERMANENT SOLUTION: Uses env_loader module for bulletproof API key loading.
    Works on: Localhost (.env), Vercel (env vars), Docker, any deployment.
    """
    
    def __init__(self):
        # BULLETPROOF: Get API key from permanent env loader
        self.api_key = get_api_key()
        self.base_url = "https://openrouter.ai/api/v1"
        self.is_configured = is_ai_enabled()
        
        if self.is_configured:
            try:
                self.client = AsyncOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=60.0,
                    default_headers={
                        "HTTP-Referer": "https://ai-job-helper-steel.vercel.app",
                        "X-Title": "CareerAgentPro",
                    }
                )
                logger.info(f"✅ OpenRouter API configured - Mode: {os.getenv('AI_MODEL_TIER', 'free')}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                self.client = None
                self.is_configured = False
        else:
            self.client = None
            logger.info("ℹ️ Running in local mode - using smart regex parsing")
            
        # Model selection: Prepare Free first
        self.free_model = "google/gemini-2.0-flash-exp:free"
        self.premium_model = "qwen/qwen-2.5-coder-32b-instruct"
        self.model = self.premium_model if os.getenv("AI_MODEL_TIER") == "premium" else self.free_model

    async def get_completion(
        self,
        prompt: str,
        system_prompt: str = "You are a professional career coach.",
        temperature: float = 0.1
    ) -> str:
        """Get AI completion from OpenRouter with enhanced error handling."""
        if not self.is_configured or not self.client:
            return '{"error": "API not configured"}'
            
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception:
            logger.exception("AI API request failed")
            return '{"error": "API request failed"}'

    async def stream_completion(
        self,
        prompt: str,
        system_prompt: str = "You are a professional career coach.",
        temperature: float = 0.1
    ):
        """Stream AI completion from OpenRouter."""
        if not self.is_configured or not self.client:
            yield '{"error": "API not configured"}'
            return

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=True
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception:
            logger.exception("AI Streaming failed")
            yield '{"error": "API request failed"}'

    async def enhance_resume(self, resume: ResumeData, job: JobDescription) -> Dict[str, Any]:
        """
        Comprehensive ATS-style resume analysis with real scoring and section-by-section improvements.
        
        Scoring factors (weighted):
        - Skills Match: 30% (keyword matching)
        - Experience Relevance: 25% (role + description keywords)
        - Keyword Density: 20% (job description terms in resume)
        - Education: 10%
        - Format Quality: 15% (completeness of sections)
        """
        
        # ============ REAL ATS SCORING ============
        scores = {
            "skills_match": 0,
            "experience_relevance": 0,
            "keyword_density": 0,
            "education": 0,
            "format_quality": 0,
        }
        
        # 1. Skills Match (30%)
        resume_skills = set(s.lower().strip() for s in (resume.skills or []))
        job_skills = set(s.lower().strip() for s in (job.skills or []))
        job_requirements = set()
        for req in (job.requirements or []):
            words = re.findall(r'\b[a-zA-Z+#]+\b', req.lower())
            job_requirements.update(words)
        
        all_job_keywords = job_skills.union(job_requirements)
        if all_job_keywords:
            matched_skills = resume_skills.intersection(all_job_keywords)
            scores["skills_match"] = min(100, int((len(matched_skills) / max(len(job_skills), 1)) * 100))
        else:
            scores["skills_match"] = 50  # Default if no job skills
        
        # 2. Experience Relevance (25%)
        experience_text = " ".join([
            f"{exp.get('role', '')} {exp.get('description', '')} {exp.get('company', '')}"
            for exp in (resume.experience or [])
        ]).lower()
        
        job_title_words = set(re.findall(r'\b[a-z]+\b', (job.title or "").lower()))
        job_desc_words = set(re.findall(r'\b[a-z]{4,}\b', (job.description or "").lower()))
        important_job_words = job_title_words.union(job_desc_words) - {'the', 'and', 'with', 'for', 'that', 'this', 'from'}
        
        if important_job_words and experience_text:
            matched_exp = sum(1 for word in important_job_words if word in experience_text)
            scores["experience_relevance"] = min(100, int((matched_exp / max(len(important_job_words), 1)) * 150))
        else:
            scores["experience_relevance"] = 30 if resume.experience else 0
        
        # 3. Keyword Density (20%)
        full_resume_text = f"{resume.summary or ''} {experience_text} {' '.join(resume.skills or [])}".lower()
        if job_skills:
            keyword_hits = sum(1 for skill in job_skills if skill in full_resume_text)
            scores["keyword_density"] = min(100, int((keyword_hits / len(job_skills)) * 100))
        else:
            scores["keyword_density"] = 50
        
        # 4. Education Score (10%)
        if resume.education and len(resume.education) > 0:
            scores["education"] = 80
            # Bonus for degree keywords
            edu_text = " ".join([f"{e.get('degree', '')} {e.get('institution', '')}" for e in resume.education]).lower()
            if any(kw in edu_text for kw in ['bachelor', 'master', 'phd', 'computer', 'engineering', 'science']):
                scores["education"] = 100
        else:
            scores["education"] = 20
        
        # 5. Format Quality (15%)
        format_checks = [
            bool(resume.name),
            bool(resume.email),
            bool(resume.phone),
            bool(resume.summary and len(resume.summary) > 50),
            len(resume.skills or []) >= 5,
            len(resume.experience or []) >= 1,
            len(resume.education or []) >= 1,
            len(resume.projects or []) >= 1,
            len(resume.certifications or []) >= 1,
        ]
        scores["format_quality"] = int((sum(format_checks) / len(format_checks)) * 100)
        
        # Calculate weighted ATS score
        ats_score = int(
            scores["skills_match"] * 0.30 +
            scores["experience_relevance"] * 0.25 +
            scores["keyword_density"] * 0.20 +
            scores["education"] * 0.10 +
            scores["format_quality"] * 0.15
        )
        ats_score = min(100, max(0, ats_score))
        
        # ============ SECTION-BY-SECTION IMPROVEMENTS ============
        section_improvements = {
            "summary": {
                "current": resume.summary or "",
                "issues": [],
                "suggested": "",
                "tips": [],
            },
            "experience": {
                "items": [],
                "general_tips": [],
            },
            "skills": {
                "matched": list(resume_skills.intersection(job_skills)) if job_skills else [],
                "missing": list(job_skills - resume_skills) if job_skills else [],
                "suggested_additions": [],
            },
            "projects": {
                "tips": [],
                "suggested": [],
            },
            "certifications": {
                "tips": [],
            }
        }
        
        # Summary improvements
        if not resume.summary:
            section_improvements["summary"]["issues"].append("Missing professional summary")
            section_improvements["summary"]["tips"].append("Add a 2-3 sentence summary highlighting your key strengths")
        elif len(resume.summary) < 100:
            section_improvements["summary"]["issues"].append("Summary is too short")
            section_improvements["summary"]["tips"].append("Expand your summary to 100-200 characters")
        
        if job.title and resume.summary and job.title.lower() not in resume.summary.lower():
            section_improvements["summary"]["tips"].append(f"Mention '{job.title}' in your summary")
        
        # Generate tailored summary
        if job.title:
            section_improvements["summary"]["suggested"] = f"Results-driven professional with expertise in {', '.join(list(resume_skills)[:3]) if resume_skills else 'relevant technologies'}. Seeking {job.title} role at {job.company or 'a leading company'} where I can leverage my experience to drive impact. {resume.summary[:200] if resume.summary else ''}"
        
        # Experience improvements
        for i, exp in enumerate(resume.experience or []):
            exp_analysis = {
                "original": exp,
                "issues": [],
                "suggested_bullets": [],
            }
            desc = exp.get("description", "")
            if len(desc) < 50:
                exp_analysis["issues"].append("Description too brief - add more details")
            if not any(char.isdigit() for char in desc):
                exp_analysis["issues"].append("Add quantified achievements (numbers, percentages, metrics)")
                exp_analysis["suggested_bullets"].append(f"• Led initiatives resulting in X% improvement in key metrics")
            if not any(action in desc.lower() for action in ['developed', 'led', 'managed', 'created', 'implemented', 'designed', 'built']):
                exp_analysis["issues"].append("Use strong action verbs (Led, Developed, Implemented, etc.)")
            
            # Suggest improved bullets based on job
            if job_skills:
                relevant = [s for s in job_skills if s not in desc.lower()][:2]
                if relevant:
                    exp_analysis["suggested_bullets"].append(f"• Utilized {', '.join(relevant)} to deliver solutions")
            
            section_improvements["experience"]["items"].append(exp_analysis)
        
        section_improvements["experience"]["general_tips"] = [
            "Start each bullet with an action verb",
            "Include metrics: numbers, percentages, dollar amounts",
            "Focus on impact and results, not just duties",
            f"Incorporate keywords: {', '.join(list(job_skills)[:5])}" if job_skills else "Add relevant technical keywords",
        ]
        
        # Skills suggestions
        section_improvements["skills"]["suggested_additions"] = list(job_skills - resume_skills)[:10] if job_skills else []
        
        if not resume.projects or len(resume.projects) == 0:
            section_improvements["projects"]["tips"].append("Add 2-3 relevant projects showcasing your skills")
        section_improvements["projects"]["tips"].append(f"Include projects using: {', '.join(list(job_skills)[:3])}" if job_skills else "Add projects with relevant technologies")
        
        # Certifications suggestions
        if not resume.certifications or len(resume.certifications) == 0:
             section_improvements["certifications"]["tips"].append("Add relevant certifications to validate your expertise")
        
        # ============ BUILD RESPONSE ============
        response = {
            "ats_score": ats_score,
            "score_breakdown": scores,
            "score": ats_score,  # Legacy support
            "feedback": self._generate_feedback(scores, ats_score),
            "tailored_summary": section_improvements["summary"]["suggested"],
            "section_improvements": section_improvements,
            "suggestions": self._generate_actionable_suggestions(section_improvements, job),
            "enhanced_resume": resume.dict(),
        }
        
        # Use AI if configured for even better suggestions
        if self.is_configured:
            try:
                ai_improvements = await self._get_ai_improvements(resume, job, section_improvements)
                if ai_improvements:
                    response["section_improvements"]["summary"]["suggested"] = ai_improvements.get("summary", response["tailored_summary"])
                    response["tailored_summary"] = ai_improvements.get("summary", response["tailored_summary"])
                    if ai_improvements.get("experience_bullets"):
                        for i, bullets in enumerate(ai_improvements["experience_bullets"]):
                            if i < len(response["section_improvements"]["experience"]["items"]):
                                response["section_improvements"]["experience"]["items"][i]["ai_suggested_bullets"] = bullets
            except Exception as e:
                logger.warning(f"AI improvements failed, using local analysis: {str(e)}")
                pass  # Use local analysis
        
        return response
    
    def _generate_feedback(self, scores: dict, total: int) -> str:
        """Generate human-readable feedback based on scores."""
        feedback = []
        if scores["skills_match"] >= 70:
            feedback.append("✓ Strong skills alignment")
        elif scores["skills_match"] >= 40:
            feedback.append("⚠ Moderate skills match - add more keywords")
        else:
            feedback.append("✗ Low skills match - update your skills section")
        
        if scores["experience_relevance"] >= 60:
            feedback.append("✓ Relevant experience detected")
        else:
            feedback.append("⚠ Tailor experience descriptions to the job")
        
        if scores["format_quality"] < 70:
            feedback.append("⚠ Complete all profile sections")
        
        return " | ".join(feedback)
    
    def _generate_actionable_suggestions(self, improvements: dict, job: JobDescription) -> List[str]:
        """Generate a list of actionable suggestions."""
        suggestions = []
        
        if improvements["skills"]["missing"]:
            suggestions.append(f"Add these skills: {', '.join(improvements['skills']['missing'][:5])}")
        
        if improvements["summary"]["issues"]:
            suggestions.append(improvements["summary"]["issues"][0])
        
        for item in improvements["experience"]["items"][:2]:
            if item["issues"]:
                suggestions.append(f"Experience: {item['issues'][0]}")
        
        suggestions.append("Quantify 2-3 achievements with specific metrics")
        
        if job.company:
            suggestions.append(f"Research {job.company} and customize your cover letter")
        
        return suggestions[:6]
    
    async def _get_ai_improvements(self, resume: ResumeData, job: JobDescription, current_improvements: dict) -> dict:
        """Use AI to generate enhanced improvements."""
        prompt = f"""As an expert career coach and ATS optimization specialist, improve this resume for the job:

JOB: {job.title} at {job.company}
Required Skills: {', '.join(job.skills[:10]) if job.skills else 'See description'}
Key Requirements: {job.description[:500] if job.description else 'N/A'}

CANDIDATE PROFILE:
- Current Summary: {resume.summary[:300] if resume.summary else 'None'}
- Skills: {', '.join(resume.skills[:15]) if resume.skills else 'None'}
- Experience: {len(resume.experience or [])} positions

Generate JSON with:
1. "summary": A powerful 2-3 sentence professional summary tailored for this exact role (include keywords: {', '.join((job.skills or [])[:5])})
2. "experience_bullets": Array of arrays - for each experience, provide 2-3 impactful bullet points with metrics

Return ONLY valid JSON, no explanation."""

        response = await self.get_completion(prompt, "You are an expert resume writer. Return valid JSON only.")
        
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse AI improvements JSON")
            return {}

    async def generate_cover_letter(self, resume: ResumeData, job: JobDescription, template_type: str) -> str:
        """Generate cover letter using Apply-Pilot style constraints."""
        name = resume.name or "Applicant"
        title = job.title or "the position"
        company = job.company or "your company"
        skills = resume.skills or []

        if self.is_configured and self.client:
            prompt = f"""
Write a 4-paragraph cover letter for a {title} role at {company}.
Constraints:
- 8-12 lines total
- 150-200 words
- No formal headers beyond the letter itself
- Minimalist, professional tone
- 4 paragraphs: Hook, Value, Alignment, CTA

Use these candidate details:
- Summary: {resume.summary or "N/A"}
- Skills: {", ".join(skills[:6]) if skills else "N/A"}
- Experience: {resume.experience[0].get("role", "") if resume.experience else ""}

Return ONLY the letter with blank lines between paragraphs.
"""
            response = await self.get_completion(prompt, "You are an expert cover letter writer.")
            cleaned = self._strip_fences(response).strip()
            if cleaned:
                return cleaned

        return self._build_cover_letter_fallback(resume, job, name, title, company)

    def _build_cover_letter_fallback(
        self,
        resume: ResumeData,
        job: JobDescription,
        name: str,
        title: str,
        company: str
    ) -> str:
        """Local fallback cover letter with Apply-Pilot constraints."""
        skills = resume.skills or []
        top_skills = ", ".join(skills[:3]) if skills else "product strategy, stakeholder alignment, and execution"
        role = resume.experience[0].get("role") if resume.experience else "a recent role"
        recent_company = resume.experience[0].get("company") if resume.experience else "a previous company"
        focus = self._infer_focus(job)
        hook_skill = ", ".join(skills[:2]) if len(skills) >= 2 else "customer outcomes and operational excellence"
        highlight = resume.summary or "delivering measurable outcomes across cross-functional teams"

        line1 = f"I was excited to see the {title} opening at {company}, especially your focus on {hook_skill}."
        line2 = (
            "The role's emphasis on measurable impact mirrors how I approach product work: "
            "clarify the problem, test quickly, and deliver outcomes."
        )
        line3 = f"In my recent role as {role} at {recent_company}, I focused on {highlight}."
        line4 = (
            "That experience sharpened my ability to translate ambiguous problems into roadmaps, "
            "align stakeholders, and deliver with clear metrics."
        )
        line5 = (
            f"I bring strengths in {top_skills} and a bias for experimentation, "
            "customer-driven decisions, and cross-functional alignment."
        )
        line6 = "I am comfortable partnering with engineering and design to ship iteratively while balancing strategy."
        line7 = f"I'd welcome the chance to discuss how I can help {company} build and scale {focus}."
        line8 = "If helpful, I'm available for a quick call and can share concrete examples of the results I've delivered."

        paragraphs = [
            f"{line1}\n{line2}",
            f"{line3}\n{line4}",
            f"{line5}\n{line6}",
            f"{line7}\n{line8}\nBest regards,\n{name}"
        ]
        return "\n\n".join(paragraphs)

    def _infer_focus(self, job: JobDescription) -> str:
        """Infer a short focus area for CTA."""
        for source in [job.about_company, job.about_job, job.description]:
            if source:
                words = source.split()
                if len(words) >= 4:
                    return " ".join(words[:4]).rstrip(".")
        return "customer experiences"

    @staticmethod
    def _strip_fences(content: str) -> str:
        if "```" in content:
            return content.split("```")[1].replace("json", "").strip()
        return content

    async def generate_communication(self, resume: ResumeData, job: JobDescription, comm_type: str) -> str:
        """Generate communication (email/LinkedIn message)."""
        name = resume.name or "Applicant"
        title = job.title or "the position"
        company = job.company or "your company"
        
        templates = {
            "email": f"""Subject: Application for {title}

Dear Hiring Manager,

I am writing to express my interest in {title} at {company}.

{resume.summary or "I have the skills and experience needed for this role."}

I would welcome the opportunity to discuss my qualifications.

Best regards,
{name}""",
            
            "linkedin_message": f"Hi! I'm interested in {title} at {company}. Would love to connect and learn more about the team!",
            
            "follow_up": f"""Subject: Following Up - {title} Application

Dear Hiring Manager,

I wanted to follow up on my application for {title} at {company}.

I remain very interested in this opportunity and am happy to provide additional information.

Best regards,
{name}"""
        }
        
        return templates.get(comm_type, templates["email"])

    def _extract_section(self, text: str, section_names: List[str], next_sections: List[str]) -> str:
        """Extract a section from resume text based on section headers."""
        pattern_start = '|'.join(section_names)
        pattern_end = '|'.join(next_sections)
        
        match = re.search(
            rf'(?:{pattern_start})\s*\n(.*?)(?=(?:{pattern_end})\s*\n|$)',
            text, re.I | re.DOTALL
        )
        return match.group(1).strip() if match else ""

    def _parse_experience(self, text: str) -> List[Dict[str, str]]:
        """Parse work experience entries from text."""
        experiences = []
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        current_exp = None
        bullets = []
        
        for i, line in enumerate(lines):
            date_match = re.search(
                r'([A-Z][a-z]{2,8}\s*\d{4}|(?:19|20)\d{2})\s*[–\-to]+\s*([A-Z][a-z]{2,8}\s*\d{4}|(?:19|20)\d{2}|Present|Current|Now)',
                line, re.I
            )
            
            if date_match:
                if current_exp and (current_exp["company"] or current_exp["role"]):
                    current_exp["description"] = " | ".join(bullets[:5])
                    experiences.append(current_exp)
                
                current_exp = {"company": "", "role": "", "duration": "", "location": "", "description": ""}
                current_exp["duration"] = f"{date_match.group(1)} - {date_match.group(2)}"
                
                role_text = line[:date_match.start()].strip()
                if role_text:
                    current_exp["role"] = role_text.rstrip(' -–|,')
                
                if i > 0:
                    prev_line = lines[i-1].strip()
                    if not prev_line.startswith(('•', '-', '○', '*')) and \
                       not any(h in prev_line.lower() for h in ['experience', 'employment', 'work history']):
                        loc_match = re.search(r',?\s*([A-Z][a-zA-Z\s]+,?\s*[A-Z]{2}|Remote|Hybrid)$', prev_line)
                        if loc_match:
                            current_exp["company"] = prev_line[:loc_match.start()].strip().rstrip(',')
                            current_exp["location"] = loc_match.group(1).strip()
                        else:
                            current_exp["company"] = prev_line
                
                bullets = []
            elif current_exp is not None:
                if line.startswith(('•', '-', '○', '*', '►')) or re.match(r'^\d+\.', line):
                    clean_line = re.sub(r'^[•\-○*►\d.]+\s*', '', line)
                    if len(clean_line) > 15:
                        bullets.append(clean_line)
        
        if current_exp and (current_exp["company"] or current_exp["role"]):
            current_exp["description"] = " | ".join(bullets[:5])
            experiences.append(current_exp)
        
        return experiences

    def _parse_education(self, text: str) -> List[Dict[str, str]]:
        """Parse education entries from text."""
        education = []
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            edu = {"institution": "", "degree": "", "graduation_year": "", "gpa": ""}
            
            is_institution = any(kw in line.lower() for kw in [
                'university', 'college', 'institute', 'school', 'academy', 'polytechnic'
            ])
            
            has_degree = bool(re.search(
                r"(Bachelor|Master|Ph\.?D|B\.?S\.?|M\.?S\.?|B\.?A\.?|M\.?A\.?|B\.?E\.?|M\.?E\.?|B\.?Tech|M\.?Tech|MBA|Associate)",
                line, re.I
            ))
            
            if is_institution or has_degree:
                if is_institution:
                    inst_match = re.match(r'^(.+?(?:University|College|Institute|School|Academy))', line, re.I)
                    if inst_match:
                        edu["institution"] = inst_match.group(1).strip()
                    else:
                        edu["institution"] = line.split(',')[0].strip()
                
                degree_text = line
                if i + 1 < len(lines):
                    degree_text = line + " " + lines[i + 1]
                
                degree_match = re.search(
                    r"\b(Bachelor(?:'s)?|Master(?:'s)?|Ph\.?D\.?|B\.S\.?|M\.S\.?|B\.A\.?|M\.A\.?|B\.E\.?|M\.E\.?|B\.?Tech|M\.?Tech|MBA|Associate(?:'s)?)\b"
                    r"(?:\s+(?:of|in)\s+)?([A-Za-z\s]{3,40})?",
                    degree_text, re.I
                )
                if degree_match:
                    degree_type = degree_match.group(1)
                    field = degree_match.group(2).strip() if degree_match.group(2) else ""
                    field = re.sub(r'\s*\d{4}.*$', '', field).strip()
                    if field and len(field) > 2:
                        edu["degree"] = f"{degree_type} in {field}"
                    else:
                        edu["degree"] = degree_type

                year_match = re.search(r'(?:19|20)\d{2}', degree_text)
                if year_match:
                    edu["graduation_year"] = year_match.group()
                
                gpa_match = re.search(r'GPA[:\s]*(\d+\.?\d*)', degree_text, re.I)
                if gpa_match:
                    edu["gpa"] = gpa_match.group(1)
                
                if edu["institution"] or edu["degree"]:
                    education.append(edu)
                    i += 1
            
            i += 1
        
        return education

    def _parse_skills(self, text: str) -> List[str]:
        """Parse skills from skills section."""
        skills = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if ':' in line:
                line = line.split(':', 1)[1]
            
            parts = re.split(r'[,|•]', line)
            for part in parts:
                skill = part.strip()
                skill = re.sub(r'\([^)]+\)', '', skill).strip()
                if skill and 2 < len(skill) < 40:
                    skills.append(skill)
        
        return list(dict.fromkeys(skills))[:25]

    def _parse_projects(self, text: str) -> List[Dict[str, str]]:
        """Parse projects from projects section."""
        projects = []
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        i = 0
        while i < len(lines):
            if lines[i].lower() == 'remote':
                i += 1
                continue
            
            date_match = re.search(r'([A-Z][a-z]{2,8}\s*\d{4})\s*[–-]\s*([A-Z][a-z]{2,8}\s*\d{4})', lines[i])
            if date_match or '(' in lines[i]:
                proj = {"name": "", "description": "", "technologies": []}
                
                name = re.sub(r'\([^)]+\)', '', lines[i])
                name = re.sub(r'[A-Z][a-z]{2,8}\s*\d{4}.*$', '', name)
                proj["name"] = name.strip()
                
                bullets = []
                i += 1
                while i < len(lines) and (lines[i].startswith('○') or lines[i].startswith('•') or lines[i].startswith('-')):
                    bullet = re.sub(r'^[○•\-]\s*', '', lines[i])
                    bullets.append(bullet)
                    i += 1
                
                proj["description"] = " | ".join(bullets[:3])
                
                if proj["name"]:
                    projects.append(proj)
                continue
            
            i += 1
        
        return projects

    async def parse_resume(self, text: str) -> Dict[str, Any]:
        """Parse resume text and extract structured data with high precision.
        
        Increased limit to 30,000 characters (~7500-10000 tokens) for full extraction.
        """
        
        result = {
            "name": "",
            "jobTitle": "",
            "email": "",
            "phone": "",
            "linkedin": "",
            "github": "",
            "website": "",
            "location": "",
            "summary": "",
            "experience": [],
            "education": [],
            "skills": [],
            "projects": [],
            "certifications": []
        }
        
        # Clean up text
        text = re.sub(r'[♂¶]', '', text)
        text = re.sub(r'obile-alt', '', text)
        text = re.sub(r'envel⌢', '', text)
        text = re.sub(r'/linkedin-in(?=linkedin)', '', text)
        
        # HEURISTIC EXTRACTION (Safety Fallback)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines[:8]:
            if '@' not in line and not any(c.isdigit() for c in line[:5]) and len(line) > 3:
                if not any(skip in line.lower() for skip in ['summary', 'skills', 'experience', 'resume', 'cv']):
                    result["name"] = line
                    break
        
        email_match = re.search(r'[\w\.\-\+]+@[\w\.\-]+\.\w+', text)
        if email_match: result["email"] = email_match.group()
        
        phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        if phone_match: result["phone"] = phone_match.group().strip()
        
        # Improved Link Extraction
        linkedin_match = re.search(r'(linkedin\.com/in/|linkedin:?\s*)([a-zA-Z0-9\-_]+)', text, re.I)
        if linkedin_match:
            username = linkedin_match.group(2)
            result["linkedin"] = username if '/' not in username else username.split('/')[-1]

        github_match = re.search(r'(github\.com/|github:?\s*)([a-zA-Z0-9\-_]+)', text, re.I)
        if github_match: result["github"] = github_match.group(2)

        portfolio_match = re.search(r'(portfolio|website):?\s*(https?://\S+|www\.\S+)', text, re.I)
        if portfolio_match: result["website"] = portfolio_match.group(2)
        
        # Extract location
        location_match = re.search(r'(Philadelphia|Pune|Boston|New York|San Francisco|Seattle|Los Angeles|Chicago|NJ|PA|CA|India|USA)[,\s]*([A-Z]{2})?\s*(\d{5})?', text, re.I)
        if location_match:
            loc_parts = [p for p in location_match.groups() if p]
            result["location"] = " ".join(loc_parts)
        
        # AI-POWERED EXTRACTION (Primary)
        if self.is_configured:
            try:
                # Use up to 25,000 characters for speed/context balance
                clipped_text = text[:25000]
                
                prompt = f"""EXTRACT RESUME DATA INTO JSON.
FAST & ACCURATE.

Output JSON only:
{{
  "name": "", "jobTitle": "", "email": "", "phone": "", "location": "",
  "linkedin": "", "github": "", "website": "", "summary": "",
  "skills": [],
  "experience": [{{"company": "", "role": "", "location": "", "duration": "", "description": ""}}],
  "education": [{{"institution": "", "degree": "", "field": "", "graduation_year": ""}}],
  "projects": [{{"name": "", "description": ""}}],
  "certifications": []
}}

RESUME TEXT:
{clipped_text}
"""
                
                response = await self.get_completion(prompt, "You are a high-speed JSON extraction engine. Return ONLY the JSON object.")
                
                # Robust extraction from markdown if AI ignores instructions
                clean_response = response.strip()
                if "```json" in clean_response:
                    clean_response = clean_response.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_response:
                    clean_response = clean_response.split("```")[1].split("```")[0].strip()
                
                # FINAL FALLBACK: Find first { and last }
                try:
                    ai_result = json.loads(clean_response)
                except json.JSONDecodeError:
                    match = re.search(r'(\{.*\})', clean_response, re.DOTALL)
                    if match:
                        ai_result = json.loads(match.group(1))
                    else:
                        raise
                
                # Smart Merge Logic
                for key, val in ai_result.items():
                    if val:
                        if key == "title":
                            result["jobTitle"] = val
                        else:
                            result[key] = val
                            
            except Exception as e:
                logger.warning(f"AI parsing failed, using heuristic results: {str(e)}")

        return result
        
    async def stream_parse_resume(self, text: str):
        """Stream resume parsing results via SSE format."""
        
        # 1. Initial status
        yield "data: " + json.dumps({"status": "extracting", "message": "Reading file content..."}) + "\n\n"
        await asyncio.sleep(0.1)

        # 2. Heuristic extraction (fast)
        result = {
            "name": "", "jobTitle": "", "email": "", "phone": "", "linkedin": "",
            "github": "", "website": "", "location": "", "summary": "",
            "experience": [], "education": [], "skills": [], "projects": [], "certifications": []
        }
        
        # Fast Link Extraction
        email_match = re.search(r'[\w\.\-\+]++@[\w\.\-]+\.\w+', text)
        if email_match: result["email"] = email_match.group()
        
        linkedin_match = re.search(r'linkedin\.com/in/([a-zA-Z0-9\-_]+)', text, re.I)
        if linkedin_match: result["linkedin"] = linkedin_match.group(1)
        
        yield "data: " + json.dumps({"status": "heuristics", "data": result}) + "\n\n"

        # 3. AI Powered Streaming Extraction
        if self.is_configured:
            yield "data: " + json.dumps({"status": "analyzing", "message": "AI is analyzing your profile nodes..."}) + "\n\n"
            
            clipped_text = text[:25000]
            prompt = f"EXTRACT RESUME DATA INTO JSON. FAST & ACCURATE.\n\nJSON SCHEMA:\n{{\"name\": \"\", \"jobTitle\": \"\", \"email\": \"\", \"phone\": \"\", \"location\": \"\", \"linkedin\": \"\", \"github\": \"\", \"website\": \"\", \"summary\": \"\", \"skills\": [], \"experience\": [], \"education\": [], \"projects\": [], \"certifications\": []}}\n\nRESUME:\n{clipped_text}"
            
            content_accumulated = ""
            async for chunk in self.stream_completion(prompt, "Return ONLY valid JSON. Accuracy is paramount.", temperature=0.0):
                content_accumulated += chunk
                # Periodically send accumulated data
                yield "data: " + json.dumps({"status": "streaming", "chunk": chunk}) + "\n\n"
            
            # Post-processing
            try:
                # Try to extract JSON if AI added markdown
                clean_json = content_accumulated.strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_json:
                    clean_json = clean_json.split("```")[1].split("```")[0].strip()
                
                final_ai_data = json.loads(clean_json)
                yield "data: " + json.dumps({"status": "completed", "data": final_ai_data}) + "\n\n"
            except Exception as e:
                logger.error(f"Stream parsing final merge failed: {e}")
                yield "data: " + json.dumps({"status": "error", "message": "Final merge failed, using partial data."}) + "\n\n"
        else:
            yield "data: " + json.dumps({"status": "completed", "data": result, "message": "AI not configured, used heuristic parsing."}) + "\n\n"
