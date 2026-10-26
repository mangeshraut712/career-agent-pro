import html as html_lib
import ipaddress
import json
import re
import socket
from html.parser import HTMLParser
from typing import List, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
import trafilatura
from bs4 import BeautifulSoup

from schemas import JobDescription
from services.ai_service import AIService

# Allowed domains for job extraction (SSRF protection)
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 3


ALLOWED_DOMAINS = frozenset([
    # Major job boards
    "linkedin.com", "www.linkedin.com", "indeed.com", "www.indeed.com",
    "glassdoor.com", "www.glassdoor.com", "monster.com", "www.monster.com",
    "ziprecruiter.com", "www.ziprecruiter.com", "careerbuilder.com", "simplyhired.com",
    "snagajob.com", "hired.com", "triplebyte.com", "toptal.com", "dice.com",
    "wellfound.com", "angel.co", "ycombinator.com", "levels.fyi", "otta.com",
    # AI Job Platforms
    "jobright.ai", "www.jobright.ai", "simplify.jobs", "www.simplify.jobs",
    # ATS platforms (Applicant Tracking Systems)
    "ashbyhq.com", "jobs.ashbyhq.com", "lever.co", "jobs.lever.co",
    "greenhouse.io", "boards.greenhouse.io", "jobvite.com", "jobs.jobvite.com",
    "smartrecruiters.com", "jobs.smartrecruiters.com", "workable.com", "apply.workable.com",
    "bamboohr.com", "www.bamboohr.com", "icims.com", "careers.icims.com",
    "myworkdayjobs.com", "workday.com", "taleo.net", "successfactors.com", "successfactors.eu",
    "recruitee.com", "personio.com", "personio.de", "teamtailor.com", "breezy.hr",
    # Tech companies
    "google.com", "amazon.jobs", "microsoft.com", "apple.com", "meta.com", "netflix.com",
    "openai.com", "anthropic.com", "nvidia.com", "tesla.com", "uber.com", "lyft.com",
    "airbnb.com", "stripe.com", "coinbase.com", "dropbox.com", "spotify.com"
])


class JobService:
    def __init__(self, ai_service: AIService):
        self.ai_service = ai_service
    
    def _is_allowed_url(self, url: str) -> bool:
        """Validate URL against allowlist to prevent SSRF attacks."""
        try:
            parsed = urlparse(url)
        except Exception:
            return False
        if parsed.scheme not in ("http", "https"):
            return False
        if parsed.username or parsed.password:
            return False
        if parsed.port not in (None, 80, 443):
            return False
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if not hostname:
            return False
        try:
            ipaddress.ip_address(hostname)
            return False
        except ValueError:
            pass
        for allowed in ALLOWED_DOMAINS:
            if hostname == allowed or hostname.endswith("." + allowed):
                return True
        return False

    def _resolves_to_public_ips(self, hostname: str) -> bool:
        """Reject allowlisted names that resolve to loopback, link-local, or private space."""
        try:
            infos = socket.getaddrinfo(hostname, None)
        except socket.gaierror:
            return False
        if not infos:
            return False
        for info in infos:
            raw_ip = info[4][0].split("%", 1)[0]
            try:
                address = ipaddress.ip_address(raw_ip)
            except ValueError:
                return False
            if not address.is_global:
                return False
        return True
    
    def _sanitize_fetch_url(self, url: str) -> Optional[str]:
        """Return an https URL that is allowlisted and resolves to public IPs.

        http is upgraded to https. Redirect targets are passed through the same
        checks so an allowed host cannot pivot to an internal address.
        """
        try:
            parsed = urlparse(url)
        except Exception:
            return None

        if parsed.scheme == "http":
            parsed = parsed._replace(scheme="https")
        if parsed.scheme != "https":
            return None
        if parsed.username or parsed.password:
            return None
        if parsed.port not in (None, 443):
            return None

        hostname = (parsed.hostname or "").lower().rstrip(".")
        if not hostname:
            return None

        # Block raw addresses and well-known non-routable names before the allowlist.
        if (hostname == "localhost" or
            hostname.startswith("127.") or
            hostname.startswith("10.") or
            hostname.startswith("192.168.") or
            hostname.startswith("169.254.") or
            hostname.startswith("0.") or
            hostname.startswith("172.16.") or
            hostname.startswith("172.17.") or
            hostname.startswith("172.18.") or
            hostname.startswith("172.19.") or
            hostname.startswith("172.2") or
            hostname.startswith("172.30.") or
            hostname.startswith("172.31.") or
            hostname == "0.0.0.0" or
            hostname == "::1"):
            return None

        is_allowed = False
        for allowed in ALLOWED_DOMAINS:
            if hostname == allowed or hostname.endswith("." + allowed):
                is_allowed = True
                break
        if not is_allowed:
            return None

        if not self._resolves_to_public_ips(hostname):
            return None

        # Reconstruct URL from validated components. Do not reuse the raw input.
        safe_url = f"https://{hostname}{parsed.path}"
        if parsed.query:
            safe_url += f"?{parsed.query}"
        return safe_url

    async def _fetch_with_httpx(self, url: str, user_agent: Optional[str] = None) -> Optional[str]:
        """Fetch URL content using httpx with SSRF protection.

        Redirects are followed manually, and every hop is revalidated. Automatic
        redirect following would let an allowlisted host point at a private address.
        """
        safe_url = self._sanitize_fetch_url(url)
        if not safe_url:
            return None

        headers = {
            "User-Agent": user_agent or "Mozilla/5.0 (compatible; JobHelper/1.0)",
            "Accept": "text/html",
        }
        try:
            async with httpx.AsyncClient(follow_redirects=False, timeout=30.0) as client:
                current = safe_url
                for hop in range(_MAX_REDIRECTS + 1):
                    response = await client.get(current, headers=headers)
                    status = getattr(response, "status_code", 200)
                    if status in _REDIRECT_STATUSES:
                        if hop == _MAX_REDIRECTS:
                            return None
                        location = ""
                        response_headers = getattr(response, "headers", None)
                        if response_headers is not None:
                            location = response_headers.get("location") or ""
                        if not location:
                            return None
                        nxt = self._sanitize_fetch_url(urljoin(current, location))
                        if not nxt:
                            return None
                        current = nxt
                        continue
                    if status >= 400:
                        return None
                    return response.text
        except httpx.HTTPError:
            return None
        except Exception:
            return None
        return None
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract readable text from HTML using trafilatura or meta tags"""
        # First try to extract from meta tags (many modern job boards use this)
        meta_content = self._extract_from_meta_tags(html_content)
        if meta_content and len(meta_content) > 200:
            return meta_content[:10000]
        
        try:
            text = trafilatura.extract(html_content, include_comments=False, include_tables=True)
            if text and len(text) > 100:
                return text[:10000]
        except Exception:
            pass
        
        # Fallback: Use html.parser (safer than regex)
        try:
            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text_parts = []
                    self.skip_tags = {'script', 'style', 'noscript'}
                    self.current_tag = None
                
                def handle_starttag(self, tag, attrs):
                    self.current_tag = tag.lower()
                
                def handle_endtag(self, tag):
                    self.current_tag = None
                
                def handle_data(self, data):
                    if self.current_tag not in self.skip_tags:
                        text = data.strip()
                        if text:
                            self.text_parts.append(text)
            
            parser = TextExtractor()
            parser.feed(html_content)
            return ' '.join(parser.text_parts)[:10000]
        except Exception:
            return meta_content or ""
    
    def _extract_from_meta_tags(self, html_content: str) -> str:
        """Extract job content from HTML - includes meta tags, LinkedIn show-more content, and structured data"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            parts = []
            extracted = {"salary": None, "location": None}
            
            # Get title from various sources
            title = None
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                title = og_title['content']
            elif soup.title:
                title = soup.title.string
            
            if title:
                parts.append(f"Job Title: {title}")
                # Extract location from LinkedIn title format "Company hiring Role in Location | LinkedIn"
                loc_match = re.search(r'in\s+([A-Z][a-zA-Z\s]+,?\s*(?:United States|[A-Z]{2})?)', title)
                if loc_match:
                    extracted["location"] = loc_match.group(1).strip()
            
            # LinkedIn specific: Get full content from show-more-less-html divs
            show_more_divs = soup.find_all(class_=re.compile(r'show-more-less-html'))
            for div in show_more_divs:
                text = div.get_text(separator=' ', strip=True)
                if len(text) > 100:
                    parts.append(html_lib.unescape(text))
            
            # LinkedIn specific: Get description from description classes
            desc_divs = soup.find_all(class_=re.compile(r'description'))
            for div in desc_divs:
                text = div.get_text(separator=' ', strip=True)
                if len(text) > 200 and text not in ''.join(parts):
                    parts.append(html_lib.unescape(text))
            
            # Get description from meta tags (fallback)
            if not parts or len(''.join(parts)) < 200:
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    parts.append(html_lib.unescape(meta_desc['content']))
                else:
                    og_desc = soup.find('meta', property='og:description')
                    if og_desc and og_desc.get('content'):
                        parts.append(html_lib.unescape(og_desc['content']))
            
            # Extract salary from HTML (LinkedIn often has this)
            salary_match = re.search(r'\$[\d,]+\s*[-–]\s*\$[\d,]+(?:\s*(?:per\s+)?(?:year|yr|annually))?', html_content, re.I)
            if salary_match:
                extracted["salary"] = salary_match.group()
                parts.append(f"Salary Range: {extracted['salary']}")
            
            # Also check for salary in structured format
            salary_patterns = [
                r'baseSalary["\']?\s*[:=]\s*["\']?\$?([\d,]+)',
                r'salary["\']?\s*[:=]\s*["\']?\$?([\d,]+)',
            ]
            for pattern in salary_patterns:
                match = re.search(pattern, html_content, re.I)
                if match and not extracted["salary"]:
                    extracted["salary"] = f"${match.group(1)}"
                    parts.append(f"Salary: {extracted['salary']}")
            
            # Extract work type (Hybrid/Remote/On-site)
            if re.search(r'\bHybrid\b', html_content, re.I):
                parts.append("Work Type: Hybrid")
            elif re.search(r'\bRemote\b', html_content, re.I):
                parts.append("Work Type: Remote")
            
            # Extract benefits directly from HTML
            benefits_found = []
            benefit_searches = [
                (r'401\(?k\)?', '401(k)'),
                (r'vision\s+insurance', 'Vision Insurance'),
                (r'disability\s+insurance', 'Disability Insurance'),
                (r'health\s+insurance', 'Health Insurance'),
                (r'dental\s+insurance', 'Dental Insurance'),
                (r'life\s+insurance', 'Life Insurance'),
                (r'paid\s+parental\s+leave', 'Paid Parental Leave'),
                (r'paid\s+time\s+off', 'Paid Time Off'),
            ]
            for pattern, name in benefit_searches:
                if re.search(pattern, html_content, re.I):
                    benefits_found.append(name)
            
            if benefits_found:
                parts.append(f"Benefits: {', '.join(benefits_found)}")
            
            # Extract location patterns
            if not extracted["location"]:
                loc_patterns = [
                    r'addressLocality["\']?\s*[:=]\s*["\']?([^"\'<,]+)',
                    r'jobLocation["\']?\s*[:=]\s*["\']?([^"\'<]+)',
                ]
                for pattern in loc_patterns:
                    match = re.search(pattern, html_content, re.I)
                    if match:
                        extracted["location"] = match.group(1).strip()
                        break
            
            if extracted["location"]:
                parts.append(f"Location: {extracted['location']}")
            
            content = '\n\n'.join(parts)
            return content[:15000] if content else ""

        except Exception as e:
            return ""



    async def extract_from_url(self, url: str) -> JobDescription:
        """Extract job details from a URL.
        
        Security: Only allowed domains can be fetched (SSRF protection).
        """
        # Security: Validate URL against allowlist
        parsed_input = urlparse(url)
        if parsed_input.scheme == "http" and parsed_input.netloc:
            url = urlunparse(parsed_input._replace(scheme="https"))

        if not self._is_allowed_url(url):
            return JobDescription(
                title="Unsupported Job Board",
                company="Unknown",
                description="This job board is not currently supported. Please copy and paste the job description manually, or use a supported job board (LinkedIn, Indeed, Glassdoor, Lever, Greenhouse, etc.).",
                requirements=["Manual entry required"],
                responsibilities=["Manual entry required"],
                url=url
            )
        
        # SECURITY: At this point, url has been validated against the allowlist.
        # We store it in safe_url to make the validation flow explicit.
        safe_url = url
        
        # Special handling for LinkedIn URLs
        original_url = url
        parsed_url = urlparse(safe_url)
        host = (parsed_url.hostname or "").lower()
        
        if host == "linkedin.com" or host.endswith(".linkedin.com"):
            # Try to extract job ID and convert to direct job view URL
            normalized_url = self._normalize_linkedin_url(safe_url)
            # Re-validate normalized URL to prevent SSRF through URL manipulation
            if self._is_allowed_url(normalized_url):
                safe_url = normalized_url
            # If normalized URL is not safe, keep the original validated URL
        
        content = None
        
        # Method 1: Try httpx first (better for meta tags and SPAs)
        # SECURITY: safe_url has been validated against our allowlist
        html_content = await self._fetch_with_httpx(safe_url)
        if html_content:
            content = self._extract_text_from_html(html_content)
        
        # Second fetch uses a browser user agent, still with redirects disabled
        # and the same allowlist. trafilatura.fetch_url follows redirects and
        # would let an allowed host pivot to an internal address.
        if not content or len(content.strip()) < 100:
            html_content = await self._fetch_with_httpx(
                safe_url,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )
            if html_content:
                extracted = self._extract_text_from_html(html_content)
                if extracted and len(extracted) > len(content or ""):
                    content = extracted
        
        # Check if we got a login page instead of job content
        if content and self._is_login_page(content):
            return JobDescription(
                title="Authentication Required",
                company=self._extract_company_from_url(original_url),
                description="This job posting requires login to view. LinkedIn and some other platforms protect job details behind authentication. Try: 1) Use the direct job link (linkedin.com/jobs/view/JOB_ID), 2) Copy and paste the job description manually.",
                requirements=["Please copy job requirements manually from the job page"],
                responsibilities=["Please copy job responsibilities manually from the job page"],
                url=original_url
            )
        
        # If still no content, return helpful message
        if not content or len(content.strip()) < 50:
            return JobDescription(
                title="Job from URL",
                company=self._extract_company_from_url(url),
                description=f"Could not extract content. Please visit: {url}",
                requirements=["See job posting for requirements"],
                responsibilities=["See job posting for responsibilities"],
                url=url
            )

        
        # Use AI to parse the content
        prompt = f"""Extract job details from the following text into JSON format.

CONTENT:
{content[:6000]}

Return ONLY valid JSON with these fields:
- title: string (job title)
- company: string (company name)
- location: string or null (work location)
- salary_range: string or null (compensation if mentioned)
- job_type: string or null (Full-time, Part-time, Contract, etc.)
- description: string (brief job description, 2-3 sentences)
- requirements: array of strings (qualifications, experience requirements)
- responsibilities: array of strings (job duties)
- skills: array of strings (required technical skills, programming languages, tools)
- benefits: array of strings (company benefits, perks if mentioned)"""
        
        try:
            response_text = await self.ai_service.get_completion(
                prompt, 
                system_prompt="You are a job data extractor. Return only valid JSON. Extract specific skills like programming languages, frameworks, and tools."
            )
            
            # Check if API returned an error
            if '"error"' in response_text:
                raise ValueError("API not available")
            
            # Extract JSON from response
            response_text = response_text.strip()
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text)
            if json_match:
                response_text = json_match.group()
            
            data = json.loads(response_text)
            
            # Verify we got actual job data
            if not data.get("title") or data.get("title") == "Unknown Title":
                raise ValueError("No title extracted")
            
            return JobDescription(
                title=str(data.get("title", "Unknown Title"))[:200],
                company=str(data.get("company", "Unknown Company"))[:200],
                location=data.get("location"),
                salary_range=data.get("salary_range"),
                job_type=data.get("job_type"),
                description=str(data.get("description", content[:500]))[:2000],
                requirements=data.get("requirements", [])[:20],
                responsibilities=data.get("responsibilities", [])[:20],
                skills=data.get("skills", [])[:30],
                benefits=data.get("benefits", [])[:15],
                url=url
            )
        except json.JSONDecodeError:
            pass
        except Exception:
            pass
        
        return self._parse_content_locally(content, url)
    
    def _parse_content_locally(self, content: str, url: str) -> JobDescription:
        """Parse job content locally using regex patterns - extracts ALL structured sections"""
        
        # Detect job source from URL
        source = self._detect_job_source(url)
        
        # Extract title
        title = self._extract_title_from_url(url) or "Job Position"
        title_patterns = [
            r"Job Title:\s*([^\n]+)",
            r"^([A-Z][a-zA-Z\s,\-]+(?:Engineer|Developer|Manager|Analyst|Designer|Lead|Director|Specialist))",
        ]
        for pattern in title_patterns:
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                title = match.group(1).strip()[:200]
                break
        
        # Clean up LinkedIn-style titles
        title = re.sub(r'\s*\|\s*LinkedIn\s*$', '', title)
        title = re.sub(r'^.*?\s+hiring\s+', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s+in\s+[A-Z][a-zA-Z\s,]+$', '', title)
        
        # Extract company
        company = self._extract_company_from_url(url)
        company_patterns = [
            r"Job Title:\s*([A-Za-z0-9\s&\.]+?)\s+(?:is\s+)?hiring",
            r"([A-Z][A-Za-z0-9\s&\.]+?)\s+(?:is\s+)?hiring",
            r"@\s*([A-Z][a-zA-Z0-9\s&]+)", 
            r"at\s+([A-Z][a-zA-Z0-9\s&]+)"
        ]
        for pattern in company_patterns:
            match = re.search(pattern, content[:2000])
            if match:
                extracted_company = match.group(1).strip()
                if 3 < len(extracted_company) < 100 and extracted_company.lower() not in ["job", "the", "we"]:
                    company = extracted_company
                    break

        # ========== EXTRACT ALL STRUCTURED SECTIONS ==========
        
        # 1. About the Job / Full Description
        about_job = self._extract_section(content, [
            r"About (?:the )?(?:job|role|position|opportunity)[:\s]*(.{100,3000}?)(?=Responsibilities|Qualifications|Requirements|What you|About|$)",
            r"(?:Job |Role )?Description[:\s]*(.{100,3000}?)(?=Responsibilities|Qualifications|Requirements|About|$)",
            r"About The Team[:\s]*(.{100,2000}?)(?=Responsibilities|$)",
        ])
        
        # 2. Responsibilities
        responsibilities = self._extract_list_section(content, [
            r"Responsibilities[:\s]*(.{50,3000}?)(?=Qualifications|Requirements|About|Benefits|Why|Preferred|$)",
            r"What you(?:'ll| will) do[:\s]*(.{50,3000}?)(?=Qualifications|Requirements|About|$)",
            r"Key Responsibilities[:\s]*(.{50,3000}?)(?=Qualifications|Requirements|$)",
        ])
        
        # 3. Minimum Qualifications
        minimum_qualifications = self._extract_list_section(content, [
            r"Minimum Qualifications[:\s]*(.{50,2000}?)(?=Preferred|About|Benefits|Why|$)",
            r"Required Qualifications[:\s]*(.{50,2000}?)(?=Preferred|Nice|About|$)",
            r"Qualifications[:\s]*(.{50,2000}?)(?=Preferred|About|Benefits|$)",
            r"Requirements[:\s]*(.{50,2000}?)(?=Preferred|About|Benefits|$)",
            r"What you(?:'ll| will) need[:\s]*(.{50,2000}?)(?=Preferred|Nice|$)",
        ])
        
        # 4. Preferred Qualifications
        preferred_qualifications = self._extract_list_section(content, [
            r"Preferred (?:Qualifications|Requirements?)[:\s]*(.{50,1500}?)(?=About|Benefits|Why|$)",
            r"Nice to have[:\s]*(.{50,1500}?)(?=About|Benefits|$)",
            r"Bonus (?:points|qualifications)[:\s]*(.{50,1500}?)(?=About|$)",
        ])
        
        # 5. About the Company
        about_company = self._extract_section(content, [
            r"About (?:the )?(?:company|us|" + re.escape(company) + r")[:\s]*(.{50,2000}?)(?=Why|Benefits|Diversity|Job Info|$)",
            r"Who we are[:\s]*(.{50,2000}?)(?=Why|Benefits|$)",
            r"Company (?:Overview|Description)[:\s]*(.{50,2000}?)(?=Why|$)",
        ])
        
        # 6. Why Join Us
        why_join = self._extract_section(content, [
            r"Why (?:join|work with) (?:us|" + re.escape(company) + r")[:\s]*(.{50,2000}?)(?=Benefits|Diversity|Job Info|$)",
            r"What we offer[:\s]*(.{50,1500}?)(?=Benefits|$)",
            r"Perks(?: and benefits)?[:\s]*(.{50,1500}?)(?=About|$)",
        ])
        
        # 7. Extract Job Info (structured metadata)
        job_info = {}
        info_patterns = [
            (r"Team[:\s]+([^\n]{5,100})", "Team"),
            (r"Reports? to[:\s]+([^\n]{5,100})", "Reports To"),
            (r"Department[:\s]+([^\n]{5,100})", "Department"),
            (r"Experience[:\s]+(\d+[^\n]{3,50})", "Experience"),
            (r"Posted[:\s]+([^\n]{5,50})", "Posted"),
            (r"Applicants?[:\s]+(\d+[^\n]{3,30})", "Applicants"),
        ]
        for pattern, key in info_patterns:
            match = re.search(pattern, content, re.I)
            if match:
                job_info[key] = match.group(1).strip()[:100]
        
        # 8. Extract Skills
        skill_keywords = [
            "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala",
            "React", "Angular", "Vue", "Next.js", "HTML", "CSS", "Tailwind", "Redux",
            "Node.js", "Express", "Django", "Flask", "FastAPI", "Spring", "Rails", ".NET",
            "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "DynamoDB",
            "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform", "Jenkins", "CI/CD", "Linux",
            "Machine Learning", "AI", "ML", "Deep Learning", "NLP", "TensorFlow", "PyTorch",
            "API", "REST", "GraphQL", "Microservices", "gRPC",
            "Git", "Agile", "Scrum", "JIRA"
        ]
        found_skills = []
        for skill in skill_keywords:
            if re.search(rf'\b{re.escape(skill)}\b', content, re.IGNORECASE):
                found_skills.append(skill)
        
        # 9. Extract Work Arrangement
        work_arrangement = None
        if re.search(r'\bHybrid\b', content, re.I):
            work_arrangement = "Hybrid"
        elif re.search(r'\b(?:remote|work\s+from\s+home|wfh)\b', content, re.I):
            work_arrangement = "Remote"
        elif re.search(r'\b(?:on-?site|in-?office)\b', content, re.I):
            work_arrangement = "On-site"
        
        # 10. Extract Job Type
        job_type = "Full-time"
        if re.search(r"\b(?:contract|contractor|temporary)\b", content, re.I): job_type = "Contract"
        elif re.search(r"\bpart[-\s]?time\b", content, re.I): job_type = "Part-time"
        elif re.search(r"\b(?:internship|interns?)\b", content, re.I): job_type = "Internship"
        
        # 11. Extract Experience Level
        experience_level = None
        if re.search(r'\b(?:senior|sr\.?|lead|principal|staff)\b', content, re.I):
            experience_level = "Senior"
        elif re.search(r'\b(?:junior|jr\.?|entry|associate)\b', content, re.I):
            experience_level = "Entry Level"
        elif re.search(r'\b(?:mid-?level|intermediate)\b', content, re.I):
            experience_level = "Mid Level"
        
        # 12. Extract Location and Salary
        location = None
        loc_patterns = [
            r"Location:\s*([A-Z][a-zA-Z\s,]+(?:United States|[A-Z]{2}))",
            r"in\s+([A-Z][a-zA-Z\s]+,?\s*(?:United States|USA|[A-Z]{2}))",
        ]
        for pattern in loc_patterns:
            match = re.search(pattern, content[:3000])
            if match:
                location = match.group(1).strip()[:100]
                break
        
        salary_range = None
        salary_match = re.search(r'\$[\d,]+\s*[-–]\s*\$[\d,]+(?:\s*(?:per\s+)?(?:year|yr|annually))?', content, re.I)
        if salary_match:
            salary_range = salary_match.group()
        
        # 13. Extract Benefits
        benefits = []
        benefits_line_match = re.search(r'Benefits:\s*([^\n]+)', content)
        if benefits_line_match:
            benefits = [b.strip() for b in benefits_line_match.group(1).split(',') if b.strip()]
        
        benefit_patterns = [
            (r'401\(?k\)?', '401(k)'),
            (r'vision\s+insurance', 'Vision Insurance'),
            (r'health\s+insurance', 'Health Insurance'),
            (r'dental\s+insurance', 'Dental Insurance'),
            (r'disability\s+insurance', 'Disability Insurance'),
            (r'life\s+insurance', 'Life Insurance'),
            (r'paid\s+(?:time\s+off|pto)', 'Paid Time Off'),
            (r'paid\s+(?:vacation|holidays)', 'Paid Vacation'),
            (r'paid\s+(?:parental\s+leave|sick)', 'Paid Leave'),
            (r'(?:stock\s+options|equity|RSUs)', 'Stock Options/Equity'),
            (r'wellness', 'Wellness Program'),
            (r'tuition\s+reimbursement', 'Tuition Reimbursement'),
        ]
        for pattern, name in benefit_patterns:
            if re.search(pattern, content, re.I):
                if name not in benefits:
                    benefits.append(name)
        
        # Combine location with work arrangement
        if location and work_arrangement:
            location = f"{location} ({work_arrangement})"
        elif work_arrangement and not location:
            location = work_arrangement
        
        # Create fallback description if no about_job
        description = about_job if about_job else content[:1000]
        
        # Combine requirements for legacy compatibility
        all_requirements = minimum_qualifications + [f"(Preferred) {q}" for q in preferred_qualifications[:5]]
        
        return JobDescription(
            title=title,
            company=company,
            location=location,
            salary_range=salary_range,
            job_type=job_type,
            work_arrangement=work_arrangement,
            experience_level=experience_level,
            about_job=about_job,
            responsibilities=responsibilities[:15],
            minimum_qualifications=minimum_qualifications[:15],
            preferred_qualifications=preferred_qualifications[:10],
            about_company=about_company,
            why_join=why_join,
            description=description[:2000],
            requirements=all_requirements[:20],
            skills=found_skills[:30],
            benefits=benefits[:12],
            job_info=job_info,
            url=url,
            source=source
        )
    
    def _extract_section(self, content: str, patterns: list) -> Optional[str]:
        """Extract a text section using multiple regex patterns"""
        for pattern in patterns:
            match = re.search(pattern, content, re.I | re.DOTALL)
            if match:
                text = match.group(1).strip()
                # Clean HTML entities
                text = re.sub(r'&[a-z]+;', ' ', text)
                text = re.sub(r'\s+', ' ', text)
                return text[:2000] if len(text) > 50 else None
        return None
    
    def _extract_list_section(self, content: str, patterns: list) -> List[str]:
        """Extract a list of items from a section"""
        for pattern in patterns:
            match = re.search(pattern, content, re.I | re.DOTALL)
            if match:
                section_text = match.group(1)
                # Extract bullet points or numbered items
                items = re.findall(r'[-•]\s*([^-•\n]{15,250})', section_text)
                if not items:
                    items = re.findall(r'\d+\.\s*([^\n]{15,250})', section_text)
                return [item.strip() for item in items if len(item.strip()) > 10][:15]
        return []
    
    def _detect_job_source(self, url: str) -> str:
        """Detect the job board source from URL using secure domain parsing"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().replace("www.", "")
            
            if domain == "linkedin.com" or domain.endswith(".linkedin.com"):
                return 'linkedin'
            elif domain == "indeed.com" or domain.endswith(".indeed.com"):
                return 'indeed'
            elif domain == "glassdoor.com" or domain.endswith(".glassdoor.com"):
                return 'glassdoor'
            elif domain == "lever.co" or domain.endswith(".lever.co"):
                return 'lever'
            elif domain == "greenhouse.io" or domain.endswith(".boards.greenhouse.io"):
                return 'greenhouse'
            elif domain == "ashbyhq.com" or domain.endswith(".ashbyhq.com"):
                return 'ashby'
            elif domain == "jobright.ai" or domain.endswith(".jobright.ai"):
                return 'jobright'
            elif domain == "simplify.jobs" or domain.endswith(".simplify.jobs"):
                return 'simplify'
            elif domain in ["wellfound.com", "angel.co"] or domain.endswith(".wellfound.com"):
                return 'wellfound'
            else:
                return 'other'
        except Exception:
            return 'other'


    def _extract_title_from_url(self, url: str) -> str:
        try:
            path = urlparse(url).path.strip("/")
            if not path: return ""
            last_part = path.split("/")[-1]
            if len(last_part) > 20 and "-" in last_part:
                return last_part.replace("-", " ").title()
            return ""
        except Exception: return ""
    
    def _extract_company_from_url(self, url: str) -> str:
        """Extract company name from URL domain securely"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            
            # Special handling for ATS subdomains
            ats_domains = ["ashbyhq.com", "lever.co", "greenhouse.io"]
            if any(ats in domain for ats in ats_domains):
                # Try to get company from path first (ATS usually format: ats.com/company/job or company.ats.com)
                path_parts = parsed.path.strip("/").split("/")
                if path_parts and path_parts[0] and len(path_parts[0]) > 2:
                     return path_parts[0].replace("-", " ").title()[:100]
            
            # Default to domain name
            company = domain.split(".")[0]
            return company.title()[:100]
        except Exception:
            return "Company"
    
    def _normalize_linkedin_url(self, url: str) -> str:
        """Convert LinkedIn URLs to the most accessible format."""
        # Extract job ID from various LinkedIn URL formats
        # Format 1: /jobs/view/123456789
        view_match = re.search(r'/jobs/view/(\d+)', url)
        if view_match:
            return f"https://www.linkedin.com/jobs/view/{view_match.group(1)}"
        
        # Format 2: currentJobId=123456789
        current_job_match = re.search(r'currentJobId=(\d+)', url)
        if current_job_match:
            return f"https://www.linkedin.com/jobs/view/{current_job_match.group(1)}"
        
        # Format 3: /jobs/collections/.../123456789
        collection_match = re.search(r'/jobs/[^/]+/(\d{8,})', url)
        if collection_match:
            return f"https://www.linkedin.com/jobs/view/{collection_match.group(1)}"
        
        return url
    
    def _is_login_page(self, content: str) -> bool:
        """Detect if the content is a login/signup page instead of job details."""
        login_indicators = [
            "Sign in",
            "sign in to your",
            "Join now",
            "Create an account",
            "email a one-time link",
            "Log in to",
            "forgot your password"
        ]
        content_lower = content.lower()
        matches = sum(1 for indicator in login_indicators if indicator.lower() in content_lower)
        # If 2+ login indicators found and no job-related content, it's a login page
        job_indicators = ["requirements", "responsibilities", "experience", "qualifications", "skills"]
        job_matches = sum(1 for indicator in job_indicators if indicator in content_lower)
        return matches >= 2 and job_matches < 2
