export type MatchResult = {
  model: string;
  semantic_similarity: number;
  semantic_score: number;
  unweighted_skill_match_score: number;
  skill_match_score: number;
  overall_score: number;
  predicted_fit_label?: string | null;
  fit_class_probabilities?: Record<string, number>;
  model_temperature?: number | null;
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
};

export type SkillEvidenceItem = {
  skill: string;
  label: "positive" | "negated" | "unclear" | string;
  confidence: number;
  sentence: string;
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
