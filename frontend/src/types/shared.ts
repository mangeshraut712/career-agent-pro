export interface JobData {
  id?: string;
  title: string;
  company: string;
  description?: string;
  requirements?: string[];
  responsibilities?: string[];
  skills?: string[];
  location?: string;
  salary?: string;
  jobType?: string;
}

export interface ProfileData {
  name: string;
  email: string;
  phone?: string;
  summary?: string;
  skills?: string[];
  experience?: Array<{
    id?: string;
    role: string;
    company: string;
    description: string;
    location?: string;
    startDate?: string;
    endDate?: string;
  }>;
  education?: Array<{ institution: string; degree: string }>;
  linkedin?: string;
  title?: string;
}
