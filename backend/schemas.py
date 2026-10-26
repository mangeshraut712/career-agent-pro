from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class JobDescription(BaseModel):
    # Basic Info
    title: str
    company: str
    location: Optional[str] = None
    salary_range: Optional[str] = None
    job_type: Optional[str] = None  # Full-time, Part-time, Contract, etc.
    work_arrangement: Optional[str] = None  # Remote, Hybrid, On-site
    experience_level: Optional[str] = None  # Entry, Mid, Senior, Lead
    posted_date: Optional[str] = None
    
    # Structured Content Sections
    about_job: Optional[str] = None  # Full job description / About the role
    responsibilities: List[str] = []  # Key responsibilities
    minimum_qualifications: List[str] = []  # Required qualifications
    preferred_qualifications: List[str] = []  # Nice to have
    about_company: Optional[str] = None  # Company description
    why_join: Optional[str] = None  # Why join us section
    
    # Legacy fields for compatibility
    description: str = ""  # Combined description (fallback)
    requirements: List[str] = []  # Combined requirements (legacy)
    
    # Skills and Benefits
    skills: List[str] = []  # Required skills
    benefits: List[str] = []  # Company benefits
    
    # Job Info Dict (for additional structured data)
    job_info: Dict[str, str] = {}  # e.g., {"Team": "Engineering", "Reports To": "VP"}
    
    # Source
    url: Optional[str] = None
    source: Optional[str] = None  # linkedin, indeed, glassdoor, etc.


class ResumeData(BaseModel):
    # Personal Info
    name: str
    email: str
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None
    
    # Career Details
    summary: str
    experience: List[dict]  # List of {company, role, duration, description}
    education: List[dict]   # List of {institution, degree, graduation_year}
    skills: List[str]
    projects: Optional[List[dict]] = None
    certifications: List[str] = []
    
    # Targeted Info (New)
    targeted_job_title: Optional[str] = None
    last_modified: Optional[str] = None
    
    # Work authorization and EEO answers are omitted unless the user supplied them.
    # Defaults such as "Male" or "authorized" would be submitted on applications.
    is_authorized_us: Optional[bool] = None
    requires_sponsorship: Optional[bool] = None
    gender: Optional[str] = None
    disability: Optional[str] = None
    is_lgbtq: Optional[str] = None
    is_veteran: Optional[str] = None
    race: Optional[str] = None
    is_hispanic: Optional[str] = None
    sexual_orientation: Optional[str] = None
    
    # Dynamic Learning (New)
    custom_questions: Optional[dict] = Field(default_factory=dict) # Key: question text, Value: answer

class EnhancementRequest(BaseModel):
    resume_data: ResumeData
    job_description: JobDescription

class EnhancementResponse(BaseModel):
    enhanced_resume: ResumeData
    feedback: str
    score: int
    tailored_summary: str

class CoverLetterRequest(BaseModel):
    resume_data: ResumeData
    job_description: JobDescription
    template_type: str = "modern"

class CoverLetterResponse(BaseModel):
    content: str

class CommunicationRequest(BaseModel):
    resume_data: ResumeData
    job_description: JobDescription
    type: str  # "email", "linkedin_message", "follow_up"


class BulletItem(BaseModel):
    id: Optional[str] = None
    text: str = Field(min_length=1, max_length=2_000)
    tags: List[str] = []
    competency: Optional[str] = None


class BulletSelectionRequest(BaseModel):
    job_description: str = Field(min_length=1, max_length=20_000)
    bullets: List[BulletItem] = Field(default_factory=list, max_length=100)
    count: int = Field(default=13, ge=1, le=100)


# ============================================
# 6-Point Bullet Framework Models
# ============================================

class SixPointBullet(BaseModel):
    """
    6-Point Bullet Framework for high-quality resume bullets.
    Every bullet must include all 6 elements.
    """
    # The 6 required points
    action: str = Field(..., min_length=1, description="Strong action verb (Led, Built, Designed, etc.)")
    context: str = Field(..., min_length=1, description="Where/what/who (cross-functional team, payment platform)")
    method: str = Field(..., min_length=1, description="How you did it (using Agile, through stakeholder interviews)")
    result: str = Field(..., min_length=1, description="Quantified outcome with metrics (reducing time by 40%)")
    impact: str = Field(..., min_length=1, description="Business effect (improving team productivity)")
    outcome: str = Field(..., min_length=1, description="Strategic value (for Fortune 500 clients)")
    
    # Metadata
    id: Optional[str] = None
    competency: Optional[str] = None  # Product Strategy, Technical, Leadership, etc.
    company_stage: Optional[str] = None  # early_stage, growth_stage, enterprise
    tags: List[str] = []
    
    # Auto-generated (read-only in API)
    full_text: Optional[str] = None
    character_count: Optional[int] = None
    has_metrics: Optional[bool] = None
    quality_score: Optional[int] = None  # 0-100
    validation_errors: List[str] = []
    validation_warnings: List[str] = []


class BulletValidationResult(BaseModel):
    """Result of bullet validation"""
    is_valid: bool
    character_count: int
    has_metrics: bool
    has_all_six_points: bool
    has_strong_verb: bool
    no_generic_language: bool
    quality_score: int  # 0-100
    
    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []
    
    # Auto-fix suggestions
    auto_fix_available: bool = False
    auto_fix_suggestions: Dict[str, str] = {}


class BulletLibraryItem(BaseModel):
    """Stored bullet in the reusable library"""
    id: str
    bullet: SixPointBullet
    created_at: Optional[str] = None
    last_used: Optional[str] = None
    usage_count: int = 0
    quality_score: int = 0
    
    # For smart selection
    competency_match_score: Optional[float] = None
    relevance_score: Optional[float] = None


class BulletSelectionCriteria(BaseModel):
    """Criteria for selecting bullets from library"""
    job_description: str
    target_competencies: List[str] = []
    company_stage: Optional[str] = None
    count: int = 13  # Total bullets needed
    distribution: Optional[List[int]] = None  # e.g., [3, 3, 3, 2, 2] for 5 roles
    
    # Preferences
    prefer_unused: bool = True
    min_quality_score: int = 70


class MetricsDetectionResult(BaseModel):
    """Result of metrics detection in text"""
    has_metrics: bool
    metrics_found: List[str] = []
    metric_types: List[str] = []  # percentage, dollar, number, ratio
    suggestions_if_missing: List[str] = []


class CompanyStageEnum(str):
    """Company stage for spinning strategy"""
    EARLY_STAGE = "early_stage"
    GROWTH_STAGE = "growth_stage"
    ENTERPRISE = "enterprise"


class SpinningRequest(BaseModel):
    """Request for industry language adaptation"""
    text: str
    target_stage: str  # early_stage, growth_stage, enterprise
    preserve_metrics: bool = True


class SpinningResult(BaseModel):
    """Result of spinning strategy"""
    original: str
    spun: str
    changes: List[Dict[str, str]] = []
    target_stage: str
    preserved_metrics: List[str] = []
    similarity: float = 1.0
    explanation: str = ""
