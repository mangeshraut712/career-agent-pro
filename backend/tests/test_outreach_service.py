"""Unit tests for Outreach Strategy Service."""

from services.outreach_service import (
    OutreachCreator,
    OutreachTrack,
    EscalationTier,
    generate_outreach_strategy,
)


def test_create_strategy_direct():
    job_data = {
        "title": "Senior Product Manager",
        "company": "Acme Health",
        "department": "Platform",
        "company_value": "patient-first innovation",
        "achievement": "Series B funding round",
    }
    resume_data = {
        "name": "Jane Doe",
        "skills": ["Product Roadmapping", "User Research", "Agile Sprints"],
        "linkedin": "https://linkedin.com/in/janedoe",
        "summary": "Experienced PM with 8+ years scaling enterprise health platforms.",
        "experience": [
            {
                "company": "HealthTech Inc",
                "description": ["Spearheaded patient intake portal reaching 250k MAU."],
            }
        ],
    }

    strategy = OutreachCreator.create_strategy(job_data, resume_data, has_mutual_connections=False)
    assert strategy.job_title == "Senior Product Manager"
    assert strategy.company == "Acme Health"
    assert strategy.primary_track == OutreachTrack.DIRECT
    assert len(strategy.messages) > 0
    assert len(strategy.target_contacts) == 5
    assert len(strategy.preparation_checklist) == 7

    # Direct track messages should contain candidate and company info
    direct_msgs = [m for m in strategy.messages if m.track == OutreachTrack.DIRECT]
    assert any("Jane Doe" in m.message for m in direct_msgs)
    assert any("Acme Health" in m.message for m in direct_msgs)


def test_create_strategy_warm_intro():
    job_data = {"title": "Staff Engineer", "company": "NovaAI"}
    resume_data = {"name": "Bob"}

    strategy = OutreachCreator.create_strategy(job_data, resume_data, has_mutual_connections=True)
    assert strategy.primary_track == OutreachTrack.WARM_INTRO


def test_create_strategy_none_and_empty():
    # Graceful handling of None
    strategy = OutreachCreator.create_strategy(None, None)
    assert strategy.job_title == "the position"
    assert strategy.company == "the company"
    assert strategy.primary_track == OutreachTrack.DIRECT
    assert len(strategy.messages) > 0
    assert len(strategy.target_contacts) > 0


def test_generate_outreach_strategy_dict_output():
    job_data = {"title": "Data Scientist", "company": "QuantLab"}
    resume_data = {
        "name": "Alice",
        "skills": "Python, SQL, PyTorch",  # Comma-separated string variant
    }

    result = generate_outreach_strategy(job_data, resume_data)
    assert isinstance(result, dict)
    assert result["job_title"] == "Data Scientist"
    assert result["company"] == "QuantLab"
    assert result["primary_track"] == "direct"
    assert isinstance(result["messages"], list)
    assert len(result["messages"]) > 0

    first_msg = result["messages"][0]
    assert "track" in first_msg
    assert "tier" in first_msg
    assert "channel" in first_msg
    assert "message" in first_msg
    assert "timing" in first_msg
    assert "tips" in first_msg


def test_subject_extraction_and_timing():
    email_template = "Subject: Hello World Application\n\nHi there!"
    subject = OutreachCreator._extract_subject(email_template)
    assert subject == "Hello World Application"

    no_subject = "Hi there, no subject line here!"
    assert OutreachCreator._extract_subject(no_subject) is None

    # Timing mappings
    assert "Immediately" in OutreachCreator._get_timing(EscalationTier.TIER_1)
    assert "3-5" in OutreachCreator._get_timing(EscalationTier.TIER_2)
    assert "7-10" in OutreachCreator._get_timing(EscalationTier.TIER_3)
