"""
Outreach Strategy Service - Multi-Track Networking with 3-Tier Escalation.
Inspired by Apply-Pilot's outreach creator agent.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OutreachTrack(Enum):
    DIRECT = "direct"           # Apply directly
    WARM_INTRO = "warm_intro"   # Through mutual connections
    COLD_OUTREACH = "cold"      # Cold LinkedIn/email
    COMMUNITY = "community"     # Newsletter, LinkedIn Post, Event, Group (G1-G4)


class CommunitySubTrack(Enum):
    G1_NEWSLETTER = "g1_newsletter"
    G2_LINKEDIN_POST = "g2_linkedin_post"
    G3_LINKEDIN_GROUP = "g3_linkedin_group"
    G4_EVENT = "g4_event"


class EscalationTier(Enum):
    TIER_1 = "tier_1"  # Initial outreach
    TIER_2 = "tier_2"  # First follow-up (3-5 days)
    TIER_3 = "tier_3"  # Final follow-up (7-10 days)


@dataclass
class OutreachMessage:
    track: OutreachTrack
    tier: EscalationTier
    channel: str  # linkedin, email
    subject: Optional[str]
    message: str
    timing: str
    tips: List[str]


@dataclass
class OutreachStrategy:
    job_title: str
    company: str
    primary_track: OutreachTrack
    messages: List[OutreachMessage]
    target_contacts: List[str]
    preparation_checklist: List[str]


class OutreachCreator:
    """Creates comprehensive multi-track outreach strategies."""
    
    # Message templates by track and tier
    TEMPLATES = {
        OutreachTrack.DIRECT: {
            EscalationTier.TIER_1: {
                "linkedin": """Hi {recruiter_name}! 👋

I just applied for the {job_title} role at {company} and wanted to reach out directly.

With my background in {key_skill}, I'm excited about the opportunity to contribute to {company_value}. My experience at {recent_company} aligns well with what you're looking for.

Would you be open to a quick chat about the role?

Best,
{candidate_name}""",
                "email": """Subject: {job_title} Application – {candidate_name}

Hi {recruiter_name},

I recently submitted my application for the {job_title} position at {company} and wanted to introduce myself.

Key highlights from my background:
• {highlight_1}
• {highlight_2}
• {highlight_3}

I'm particularly drawn to {company} because of {company_value}. I'd welcome the opportunity to discuss how my experience could contribute to your team.

Thank you for your consideration.

Best regards,
{candidate_name}
{linkedin_url}"""
            },
            EscalationTier.TIER_2: {
                "email": """Subject: Following Up – {job_title} Application

Hi {recruiter_name},

I wanted to follow up on my application for the {job_title} role submitted last week.

I remain very enthusiastic about this opportunity and believe my experience in {key_skill} would be valuable to the team.

Is there any additional information I can provide to support my application?

Best regards,
{candidate_name}"""
            },
            EscalationTier.TIER_3: {
                "email": """Subject: Final Check-In – {job_title} Position

Hi {recruiter_name},

I wanted to check in one more time regarding the {job_title} position.

I understand you're likely reviewing many candidates, but I wanted to reiterate my strong interest in {company} and this role specifically.

If this position has been filled or moved in a different direction, I'd still love to stay connected for future opportunities.

Best regards,
{candidate_name}"""
            }
        },
        OutreachTrack.WARM_INTRO: {
            EscalationTier.TIER_1: {
                "linkedin": """Hi {connection_name}!

Hope you're doing well! I noticed you're connected to people at {company} – I'm very interested in their {job_title} role.

Would you be comfortable making an introduction, or sharing any insights about the team?

Happy to chat more about what I'm looking for if helpful!

Thanks,
{candidate_name}"""
            }
        },
        OutreachTrack.COLD_OUTREACH: {
            EscalationTier.TIER_1: {
                "linkedin": """Hi {contact_name}!

I came across {company}'s {job_title} role and was impressed by {company_achievement}. Your work on {their_project} particularly caught my attention.

I have a background in {key_skill} and would love to learn more about the team and role. Would you have 15 minutes for a quick call?

Best,
{candidate_name}""",
                "email": """Subject: Quick Question About {job_title} at {company}

Hi {contact_name},

I noticed {company}'s {job_title} opening and, given your role as {their_role}, thought you might have valuable insights.

I'm particularly interested in:
• The team structure and culture
• Key challenges the role addresses
• What success looks like in the first 90 days

Would you have 15 minutes for a brief call this week?

Best,
{candidate_name}"""
            }
        },
        OutreachTrack.COMMUNITY: {
            CommunitySubTrack.G1_NEWSLETTER: {
                "linkedin": """Hi {contact_name}! I saw you were mentioned in {newsletter_name} regarding the {job_title} role at {company}. I've been following your work on {their_project} and would love to connect and learn more about the team!""",
                "email": """Subject: From {newsletter_name}: Interest in {job_title} at {company}
 
 Hi {contact_name},
 
 I'm reaching out after seeing the {job_title} role mentioned in {newsletter_name}. I've been a long-time reader and noticed {company}'s focus on {company_value}, which aligns perfectly with my background in {key_skill}.
 
 I'd love to learn about the PM culture at {company} and what you enjoy most about the team.
 
 Best,
 {candidate_name}"""
            },
            CommunitySubTrack.G2_LINKEDIN_POST: {
                "linkedin": """Hi {contact_name}! I really enjoyed your recent post about {post_topic}. I noticed {company} is hiring for a {job_title}—given my background in {key_skill}, I'd love to connect and learn more!"""
            },
            CommunitySubTrack.G3_LINKEDIN_GROUP: {
                "linkedin": """Hi {contact_name}! We're both members of the {group_name} group. I saw {company} is hiring for a {job_title} and would love to connect with a fellow group member to learn about the team!"""
            },
            CommunitySubTrack.G4_EVENT: {
                "linkedin": """Hi {contact_name}! It was great meeting you at {event_name}. I'm very interested in the {job_title} role at {company} and would love to connect and follow up on our conversation!"""
            }
        }
    }
    
    TARGET_CONTACT_TYPES = [
        "Hiring Manager (direct supervisor for the role)",
        "Recruiter or Talent Acquisition",
        "Team members in similar roles",
        "Department head or VP",
        "Mutual connections at the company"
    ]
    
    @classmethod
    def create_strategy(
        cls,
        job_data: Optional[Dict[str, Any]] = None,
        resume_data: Optional[Dict[str, Any]] = None,
        has_mutual_connections: bool = False
    ) -> OutreachStrategy:
        """Create a comprehensive outreach strategy."""
        job_data = job_data or {}
        resume_data = resume_data or {}
        
        job_title = job_data.get("title", "the position")
        company = job_data.get("company", "the company")
        candidate_name = resume_data.get("name", "Candidate")
        
        # Determine primary track
        primary_track = OutreachTrack.WARM_INTRO if has_mutual_connections else OutreachTrack.DIRECT
        
        # Generate messages for all tracks
        messages = cls._generate_messages(job_data, resume_data, primary_track)
        
        # Get target contacts
        target_contacts = cls._get_target_contacts(job_data)
        
        # Generate preparation checklist
        checklist = cls._get_preparation_checklist(job_data, resume_data)
        
        return OutreachStrategy(
            job_title=job_title,
            company=company,
            primary_track=primary_track,
            messages=messages,
            target_contacts=target_contacts,
            preparation_checklist=checklist
        )
    
    @classmethod
    def _generate_messages(
        cls,
        job_data: Dict[str, Any],
        resume_data: Dict[str, Any],
        primary_track: OutreachTrack
    ) -> List[OutreachMessage]:
        """Generate all outreach messages."""
        messages = []
        
        # Context for template filling
        context = cls._build_context(job_data, resume_data)
        
        # Direct track - all tiers
        for tier, channels in cls.TEMPLATES[OutreachTrack.DIRECT].items():
            for channel, template in channels.items():
                messages.append(OutreachMessage(
                    track=OutreachTrack.DIRECT,
                    tier=tier,
                    channel=channel,
                    subject=cls._extract_subject(template),
                    message=cls._fill_template(template, context),
                    timing=cls._get_timing(tier),
                    tips=cls._get_tips(OutreachTrack.DIRECT, tier, channel)
                ))
        
        # Cold outreach - tier 1
        for channel, template in cls.TEMPLATES[OutreachTrack.COLD_OUTREACH][EscalationTier.TIER_1].items():
            messages.append(OutreachMessage(
                track=OutreachTrack.COLD_OUTREACH,
                tier=EscalationTier.TIER_1,
                channel=channel,
                subject=cls._extract_subject(template),
                message=cls._fill_template(template, context),
                timing="After researching the contact",
                tips=cls._get_tips(OutreachTrack.COLD_OUTREACH, EscalationTier.TIER_1, channel)
            ))
        
        return messages
    
    @classmethod
    def _build_context(cls, job_data: Optional[Dict[str, Any]] = None, resume_data: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Build context for template filling."""
        job_data = job_data or {}
        resume_data = resume_data or {}
        skills = resume_data.get("skills", [])
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",") if s.strip()]
        elif not isinstance(skills, list):
            skills = []
        
        experience = resume_data.get("experience", [])
        recent_company = "my previous company"
        if isinstance(experience, list) and experience and isinstance(experience[0], dict):
            recent_company = experience[0].get("company", "my previous company")
        
        highlights = cls._extract_highlights(resume_data)
        
        return {
            "job_title": job_data.get("title", "the position"),
            "company": job_data.get("company", "your company"),
            "candidate_name": resume_data.get("name", ""),
            "key_skill": skills[0] if skills else "relevant skills",
            "recent_company": recent_company,
            "company_value": job_data.get("company_value", "your mission and growth"),
            "company_achievement": job_data.get("achievement", "your recent growth"),
            "highlight_1": highlights[0] if len(highlights) > 0 else "Strong technical background",
            "highlight_2": highlights[1] if len(highlights) > 1 else "Proven track record of delivery",
            "highlight_3": highlights[2] if len(highlights) > 2 else "Excellent collaboration skills",
            "linkedin_url": resume_data.get("linkedin", ""),
            "recruiter_name": "Hiring Team",
            "connection_name": "[Connection Name]",
            "contact_name": "[Contact Name]",
            "their_role": "[Their Role]",
            "their_project": "[Their Notable Project]"
        }
    
    @classmethod
    def _extract_highlights(cls, resume_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """Extract 3 key highlights from resume."""
        resume_data = resume_data or {}
        highlights = []
        
        # From summary
        summary = str(resume_data.get("summary") or "")
        if summary:
            highlights.append(summary[:100] + "..." if len(summary) > 100 else summary)
        
        # From experience
        exp_list = resume_data.get("experience", [])
        if isinstance(exp_list, list):
            for exp in exp_list[:2]:
                if not isinstance(exp, dict):
                    continue
                desc = exp.get("description", "")
                if isinstance(desc, list) and desc:
                    first_desc = str(desc[0])
                    highlights.append(first_desc[:80] + "..." if len(first_desc) > 80 else first_desc)
                elif desc:
                    desc_str = str(desc)
                    highlights.append(desc_str[:80] + "..." if len(desc_str) > 80 else desc_str)
        
        return highlights[:3]
    
    @classmethod
    def _fill_template(cls, template: str, context: Dict[str, str]) -> str:
        """Fill template with context values."""
        result = template
        for key, value in context.items():
            result = result.replace("{" + key + "}", value)
        return result
    
    @classmethod
    def _extract_subject(cls, template: str) -> Optional[str]:
        """Extract subject line if present."""
        if template.startswith("Subject:"):
            first_line = template.split("\n")[0]
            return first_line.replace("Subject:", "").strip()
        return None
    
    @classmethod
    def _get_timing(cls, tier: EscalationTier) -> str:
        """Get recommended timing for each tier."""
        timings = {
            EscalationTier.TIER_1: "Immediately after applying",
            EscalationTier.TIER_2: "3-5 business days after initial outreach",
            EscalationTier.TIER_3: "7-10 business days after first follow-up"
        }
        return timings.get(tier, "When appropriate")
    
    @classmethod
    def _get_tips(cls, track: OutreachTrack, tier: EscalationTier, channel: str) -> List[str]:
        """Get tips for specific outreach type."""
        tips = {
            (OutreachTrack.DIRECT, EscalationTier.TIER_1, "linkedin"): [
                "Personalize with something specific about their work",
                "Keep it under 100 words for LinkedIn",
                "Send Tuesday-Thursday, 9-11am local time"
            ],
            (OutreachTrack.DIRECT, EscalationTier.TIER_1, "email"): [
                "Use a clear, specific subject line",
                "Include your LinkedIn profile",
                "Reference specific job requirements"
            ],
            (OutreachTrack.COLD_OUTREACH, EscalationTier.TIER_1, "linkedin"): [
                "Research the person's background first",
                "Mention something specific about their work",
                "Ask a genuine question, not just 'can I have a job'"
            ]
        }
        return tips.get((track, tier, channel), [
            "Be professional and concise",
            "Show genuine interest in the company",
            "Make it easy to respond"
        ])
    
    @classmethod
    def _get_target_contacts(cls, job_data: Dict[str, Any]) -> List[str]:
        """Get list of target contact types to find."""
        company = job_data.get("company", "the company")
        return [
            f"Hiring Manager for {job_data.get('title', 'the role')} at {company}",
            f"Recruiter or TA at {company}",
            f"Team members in {job_data.get('department', 'the department')}",
            f"Mutual connections at {company}",
            f"Alumni from your school at {company}"
        ]
    
    @classmethod
    def _get_preparation_checklist(cls, job_data: Dict[str, Any], resume_data: Dict[str, Any]) -> List[str]:
        """Generate preparation checklist."""
        return [
            f"✅ Update resume for {job_data.get('company', 'company')}'s requirements",
            "✅ Research company news and recent achievements",
            "✅ Find 3-5 contact targets on LinkedIn",
            "✅ Prepare talking points for each target",
            "✅ Set calendar reminders for follow-ups",
            "✅ Track all outreach in a spreadsheet",
            f"✅ Prepare answers for common {job_data.get('title', 'role')} interview questions"
        ]


def generate_outreach_strategy(
    job_data: Optional[Dict[str, Any]] = None,
    resume_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """API convenience function."""
    strategy = OutreachCreator.create_strategy(job_data or {}, resume_data or {})
    
    return {
        "job_title": strategy.job_title,
        "company": strategy.company,
        "primary_track": strategy.primary_track.value,
        "target_contacts": strategy.target_contacts,
        "preparation_checklist": strategy.preparation_checklist,
        "messages": [
            {
                "track": m.track.value,
                "tier": m.tier.value,
                "channel": m.channel,
                "subject": m.subject,
                "message": m.message,
                "timing": m.timing,
                "tips": m.tips
            }
            for m in strategy.messages
        ]
    }
