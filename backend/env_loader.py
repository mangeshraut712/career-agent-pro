"""
PERMANENT SOLUTION: Environment Variable Configuration
Works for: Localhost (.env), Vercel, Docker, Any deployment

This module ensures OPENROUTER_API_KEY is ALWAYS loaded correctly.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ============================================================================
# STEP 1: Load .env file for LOCAL DEVELOPMENT
# ============================================================================
# This finds .env file in backend directory or project root
env_path = Path(__file__).parent / '.env'
if not env_path.exists():
    env_path = Path(__file__).parent.parent / '.env'

if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"✅ Loaded environment from: {env_path}")
else:
    print("ℹ️  No .env file found - using system environment variables")

# ============================================================================
# STEP 2: Get OPENROUTER_API_KEY with multiple fallback strategies
# ============================================================================

def get_openrouter_api_key() -> str:
    """
    Get OpenRouter API key with bulletproof fallback strategy.
    
    Priority order:
    1. os.environ['OPENROUTER_API_KEY'] (Vercel, Docker, System env)
    2. os.getenv('OPENROUTER_API_KEY') (Fallback)
    3. Check .env file directly
    4. Empty string (will use non-AI mode)
    
    Returns:
        str: API key or empty string
    """
    # Try 1: Direct environment access (Vercel sets this)
    api_key = os.environ.get('OPENROUTER_API_KEY', '')
    if api_key:
        print("✅ OPENROUTER_API_KEY loaded from environment")
        return api_key
    
    # Try 2: Alternative  environment variable name
    api_key = os.environ.get('OPENROUTER_KEY', '')
    if api_key:
        print("✅ OPENROUTER_KEY loaded from environment")
        return api_key
    
    # Try 3: Read .env file directly as fallback
    if env_path.exists():
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    if line.strip().startswith('OPENROUTER_API_KEY='):
                        api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                        if api_key:
                            print("✅ OPENROUTER_API_KEY loaded from .env file")
                            return api_key
        except Exception as e:
            print(f"⚠️  Could not read .env file: {e}")
    
    print("⚠️  OPENROUTER_API_KEY not found - AI features will use fallback mode")
    return ''


def validate_api_key(api_key: str) -> bool:
    """
    Validate that API key looks correct.
    
    OpenRouter keys typically:
    - Start with 'sk-or-v1-'
    - Are 64+ characters long
    
    Args:
        api_key: The API key to validate
        
    Returns:
        bool: True if key looks valid
    """
    if not api_key:
        return False
    
    # Basic validation
    if len(api_key) < 20:
        print("⚠️  API key too short")
        return False
    
    # OpenRouter keys usually start with specific prefix
    if api_key.startswith('sk-or-v1-'):
        print(f"✅ API key format looks correct (OpenRouter format)")
        return True
    elif api_key.startswith('sk-'):
        print(f"✅ API key format looks valid (generic format)")
        return True
    else:
        print(f"⚠️  API key doesn't match expected format, but will try anyway")
        return True  # Still return True to allow usage


# ============================================================================
# STEP 3: Initialize API key on module import
# ============================================================================

# Get the API key
OPENROUTER_API_KEY = get_openrouter_api_key()

# Validate it
IS_API_KEY_VALID = validate_api_key(OPENROUTER_API_KEY)

# Export status
if OPENROUTER_API_KEY and IS_API_KEY_VALID:
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  🚀 AI MODE ENABLED                                      ║
║  ✅ OpenRouter API Key: Configured                      ║
║  ✅ AI Features: ACTIVE                                 ║
║  ✅ Model: qwen/qwen-2.5-coder-32b-instruct             ║
╚══════════════════════════════════════════════════════════╝
""")
else:
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  ⚙️  FALLBACK MODE                                       ║
║  ℹ️  OpenRouter API Key: Not found                      ║
║  ✅ AI Features: Using heuristic fallbacks              ║
║  ℹ️  Set OPENROUTER_API_KEY to enable full AI          ║
╚══════════════════════════════════════════════════════════╝
""")


# ============================================================================
# STEP 4: Helper functions for services
# ============================================================================

def get_api_key() -> str:
    """Get the loaded API key."""
    return OPENROUTER_API_KEY


def is_ai_enabled() -> bool:
    """Check if AI mode is enabled."""
    return bool(OPENROUTER_API_KEY and IS_API_KEY_VALID)


def get_ai_status() -> dict:
    """Get comprehensive AI status."""
    return {
        "ai_enabled": is_ai_enabled(),
        "api_key_present": bool(OPENROUTER_API_KEY),
        "api_key_valid": IS_API_KEY_VALID,
        "mode": "AI" if is_ai_enabled() else "Fallback",
        "model": "qwen/qwen-2.5-coder-32b-instruct" if is_ai_enabled() else "Heuristic",
    }
