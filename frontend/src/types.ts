export type MatchResult = {
  model: string;
  semantic_similarity: number;
  semantic_score: number;
  unweighted_skill_match_score: number;
  skill_match_score: number;
  overall_score: number;
  resume_skills: string[];
  job_skills: string[];
  matched_skills: string[];
  missing_skills: string[];
  extra_resume_skills: string[];
  match_category: string;
  hr_evaluation: string;
  interview_focus: string[];
  candidate_suggestions: string[];
};

export type ModelComparisonItem = {
  model_id: string;
  model_label: string;
  description: string;
  model: string;
  available: boolean;
  semantic_score?: number;
  unweighted_skill_match_score?: number;
  skill_match_score?: number;
  overall_score?: number;
  match_category?: string;
  error?: string;
};

export type ModelComparisonPayload = {
  items: ModelComparisonItem[];
};

export type SamplePayload = {
  resume_text: string;
  job_description_text: string;
};

export type HealthPayload = {
  status: string;
  model: string;
  model_path: string;
};
