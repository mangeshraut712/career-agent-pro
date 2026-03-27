<div align="center">

# 🚀 CareerAgentPro

### AI-powered job search, resume tailoring, and career coaching in one workspace

[![Live Demo](https://img.shields.io/badge/Live_Demo-ai--job--helper--steel.vercel.app-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://ai-job-helper-steel.vercel.app)
[![Next.js](https://img.shields.io/badge/Next.js-16.1-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-blue?style=for-the-badge&logo=typescript)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[Features](#-features) • [Stack](#-stack) • [Quick Start](#-quick-start) • [Structure](#-project-structure) • [Scripts](#-scripts) • [License](#-license) • [Contact](#-contact)

</div>

---

## Table of Contents

- [About](#-about)
- [Features](#-features)
- [Stack](#-stack)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Scripts](#-scripts)
- [License](#-license)
- [Contact](#-contact)

## About

CareerAgentPro combines a Next.js 16 frontend with a FastAPI backend to automate the parts of job hunting that take the most time: parsing resumes, matching jobs, tailoring content, and drafting outreach. The app includes streaming AI responses, a polished dashboard, and a set of workflow-specific tools for resumes, cover letters, and interview prep.

## Features

- Resume parsing from PDF, DOCX, and TXT uploads
- Job URL extraction and fit analysis
- AI resume enhancement with ATS-oriented feedback
- Cover letter generation and multi-track outreach drafts
- Bullet building, competency scoring, and resume verification
- Real-time SSE streaming for long-running AI tasks
- Dashboards for profile, jobs, resumes, communication, outreach, and interview prep

## Stack

| Area | Technologies |
| --- | --- |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4, Framer Motion |
| Backend | FastAPI, Python 3.12, Pydantic, SSE Starlette, LangChain |
| AI & Data | OpenRouter, structured outputs, PyPDF2, python-docx |
| Tooling | Vercel, npm, pytest, ESLint |

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- An OpenRouter API key

### Run the backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

### Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open the app at `http://localhost:3000` and the API health check at `http://localhost:8000/health`.

## Project Structure

```text
career-agent-pro/
├── backend/          # FastAPI app, routes, services, and tests
├── frontend/         # Next.js App Router UI and design system
├── sample_resume.txt # Local sample input for testing flows
├── vercel.json       # Vercel deployment config
└── README.md         # Project overview and setup
```

## Scripts

Frontend commands live in `frontend/package.json`:

```bash
npm run dev
npm run build
npm run start
npm run lint
npm run type-check
npm run format
```

Backend development uses the FastAPI app directly:

```bash
uvicorn main:app --reload --port 8000
python -m pytest
```

## License

MIT. See [LICENSE](LICENSE) for details.

## Contact

- Live demo: [ai-job-helper-steel.vercel.app](https://ai-job-helper-steel.vercel.app)
- Repository: [mangeshraut712/career-agent-pro](https://github.com/mangeshraut712/career-agent-pro)
- Issues: open a GitHub issue in this repository