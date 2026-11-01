"use client";

import { useState, useEffect } from "react";
import { secureGet, secureSet } from "@/lib/secureStorage";
import { STORAGE_KEYS } from "@/lib/storageKeys";
import { JobData, ProfileData } from "@/types/shared";

export function useAppData() {
  const [currentJob, setCurrentJob] = useState<JobData | null>(null);
  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadData = () => {
      const savedJob = secureGet<JobData>(STORAGE_KEYS.CURRENT_JOB_FOR_RESUME);
      if (savedJob) setCurrentJob(savedJob);

      const savedProfile = secureGet<ProfileData>(STORAGE_KEYS.PROFILE);
      if (savedProfile) setProfile(savedProfile);

      setIsLoading(false);
    };
    loadData();
  }, []);

  const updateProfile = (newProfile: ProfileData) => {
    setProfile(newProfile);
    secureSet(STORAGE_KEYS.PROFILE, newProfile);
  };

  return { currentJob, profile, isLoading, updateProfile };
}
