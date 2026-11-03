# Contributing to CareerAgentPro

Thank you for your interest in contributing to CareerAgentPro! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)

---

## 🤝 Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [mbr63@drexel.edu].

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

---

## 🚀 Getting Started

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/career-agent-pro.git
   cd career-agent-pro
   ```

### Setup Development Environment

#### Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

#### Frontend Setup
```bash
cd frontend
npm install
```

### Run Development Servers
```bash
# Terminal 1 - Backend
cd backend
source venv/bin/activate
uvicorn main:app --reload

# Terminal 2 - Frontend
cd frontend
npm run dev
```

---

## 💻 Development Workflow

### 1. Create a Branch

Create a branch for your feature or bugfix:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Adding tests
- `chore/` - Maintenance tasks

### 2. Make Changes

- Write clean, readable code
- Follow existing code style
- Add tests for new features
- Update documentation as needed
- Keep commits atomic and focused

### 3. Test Your Changes

```bash
# Backend tests
cd backend
pytest

# Frontend linting
cd frontend
npm run lint

# Frontend type check
npm run type-check

# Frontend build test
npm run build
```

### 4. Commit Your Changes

Follow commit message guidelines (see below)

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

---

## 📝 Coding Standards

### Python (Backend)

- **Style Guide**: Follow [PEP 8](https://pep8.org/)
- **Formatting**: Use `black` for code formatting
- **Linting**: Use `flake8` for linting
- **Type Hints**: Use type hints for all function parameters and return values
- **Docstrings**: Use Google-style docstrings

Example:
```python
def enhance_resume(resume: dict, job: dict) -> dict:
    \"\"\"
    Enhance resume based on job requirements.
    
    Args:
        resume: User's resume data
        job: Target job data
        
    Returns:
        dict: Enhanced resume with suggestions
        
    Raises:
        ValueError: If resume or job data is invalid
    \"\"\"
    # Implementation
```

### TypeScript/React (Frontend)

- **Style Guide**: Follow [Airbnb React/JSX Style Guide](https://github.com/airbnb/javascript/tree/master/react)
- **Formatting**: Use Prettier (run `npm run format`)
- **Linting**: ESLint is configured (run `npm run lint`)
- **Types**: Always use TypeScript, avoid `any` types
- **Components**: Use functional components with hooks

Example:
```typescript
interface ProfileData {
    name: string;
    email: string;
    skills: string[];
}

export function ProfileCard({ profile }: { profile: ProfileData }) {
    return (
        <AppleCard>
            <h3>{profile.name}</h3>
            <p>{profile.email}</p>
        </AppleCard>
    );
}
```

### General Principles

- **DRY**: Don't Repeat Yourself
- **KISS**: Keep It Simple, Stupid
- **YAGNI**: You Aren't Gonna Need It
- **Single Responsibility**: Each function/class should do one thing well
- **Consistent Naming**: Use clear, descriptive names

---

## 📜 Commit Guidelines

### Commit Message Format

```
type(scope): subject

body (optional)

footer (optional)
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semi-colons, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

### Examples

```bash
feat(resume): add LaTeX export functionality

Add ability to export resumes in LaTeX format for academic use.
Includes template selection and custom formatting options.

Closes #123
```

```bash
fix(api): handle null response from OpenRouter

Add proper error handling when OpenRouter API returns null.
Implement graceful fallback to heuristic mode.
```

```bash
docs(readme): update installation instructions

Clarify Python version requirement and add troubleshooting section.
```

---

## 🔄 Pull Request Process

### Before Submitting

1. ✅ Ensure all tests pass
2. ✅ Update documentation if needed
3. ✅ Add tests for new features
4. ✅ Follow code style guidelines
5. ✅ Rebase on latest main branch
6. ✅ Write clear PR description

### PR Description Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How to test these changes

## Screenshots (if applicable)
Add screenshots for UI changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tested locally
```

### Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged
4. Delete your branch after merge

---

## 🧪 Testing

### Backend Testing

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_ai_service.py

# Run specific test
pytest tests/test_ai_service.py::test_enhance_resume
```

### Frontend Testing

```bash
cd frontend

# Lint check
npm run lint

# Type check
npm run type-check

# Build test
npm run build
```

### Manual Testing Checklist

- [ ] Upload resume - parsing works
- [ ] Extract job from URL
- [ ] Get AI suggestions
- [ ] Generate cover letter
- [ ] Export to PDF/DOCX
- [ ] Test dark mode
- [ ] Test responsive design
- [ ] Test error handling

---

## 📚 Documentation

### When to Update Documentation

- Adding new features
- Changing existing features
- Adding new API endpoints
- Modifying environment variables
- Changing deployment process

### Documentation Files

- `README.md` - Main project documentation
- Code comments - Inline documentation
- Docstrings - Function/class documentation
- API docs - Endpoint documentation

### Documentation Style

- Use clear, concise language
- Include code examples
- Add screenshots for UI features
- Keep it up to date
- Link to external resources when relevant

---

## 🐛 Bug Reports

### How to Report a Bug

1. Check if the bug is already reported
2. Create a new issue with:
   - Clear title
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots if applicable
   - Environment details (OS, Python version, Node version)
   - Error logs if available

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g. macOS, Windows, Linux]
- Python version: [e.g. 3.12]
- Node version: [e.g. 18.0]
- Browser: [e.g. Chrome, Safari]

**Additional context**
Any other context about the problem.
```

---

## 💡 Feature Requests

### How to Request a Feature

1. Check if feature is already requested
2. Create a new issue with:
   - Clear title
   - Detailed description
   - Use case/motivation
   - Possible implementation ideas
   - Alternative solutions considered

---

## 🏷️ Labels

Issues and PRs are labeled to help organization:

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Documentation improvements
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `question` - Further information requested
- `wontfix` - Will not be worked on

---

## 📞 Getting Help

- 💬 **GitHub Discussions**: Ask questions
- 🐛 **GitHub Issues**: Report bugs
- 📖 **Documentation**: Read the README
- 📧 **Email**: [mbr63@drexel.edu]

---

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors page

Thank you for contributing to CareerAgentPro! 🎉

---

**Happy Coding!** 🚀
