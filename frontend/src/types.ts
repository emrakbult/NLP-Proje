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
  negated_resume_skills: string[];
  unclear_resume_skills: string[];
  skill_evidence: SkillEvidenceItem[];
  skill_evidence_model_available: boolean;
  match_category: string;
  hr_evaluation: string;
  interview_focus: string[];
  candidate_suggestions: string[];
};

export type SkillEvidenceItem = {
  skill: string;
  label: "positive" | "negated" | "unclear" | string;
  confidence: number;
  sentence: string;
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
  skill_evidence_classifier_available: boolean;
  skill_evidence_classifier_path: string;
};

export type ExtractedDocumentPayload = {
  file_name: string;
  extracted_text: string;
  character_count: number;
  converter: string;
};
