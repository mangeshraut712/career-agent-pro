export interface Experience {
  id: string;
  company: string;
  role: string;
  location: string;
  startDate: string;
  endDate: string;
  type: string;
  description: string;
}

export interface Education {
  id: string;
  institution: string;
  degree: string;
  field: string;
  startDate: string;
  endDate: string;
}

export interface Project {
  id: string;
  name: string;
  description: string;
  technologies?: string[];
}

export interface ProfileData {
  name: string;
  title: string;
  email: string;
  phone: string;
  location: string;
  linkedin: string;
  github: string;
  portfolio: string;
  jobSearchStatus: string;
  experience: Experience[];
  education: Education[];
  projects: Project[];
  certifications: string[];
  skills: string[];
  preferredSkills: string[];
  summary: string;
}
