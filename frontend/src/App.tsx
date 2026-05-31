import { useEffect, useMemo, useState } from "react";
import { analyzeMatch, compareModels, extractText, getHealth, getSample } from "./api";
import type { ExtractedDocumentPayload, HealthPayload, MatchResult, ModelComparisonItem } from "./types";

const pipelineSteps = [
  "Encoder embeddings",
  "Cosine similarity",
  "Skill extraction",
  "Explainable HR notes"
];

type UploadTarget = "resume" | "job";
type UploadInfo = ExtractedDocumentPayload | null;

function formatNumber(value: number): string {
  return value.toFixed(2);
}

function scoreTone(score: number): "good" | "warning" | "bad" {
  if (score >= 75) return "good";
  if (score >= 45) return "warning";
  return "bad";
}

function ScoreCard({
  title,
  value,
  accent = scoreTone(value)
}: {
  title: string;
  value: number;
  accent?: "good" | "warning" | "bad" | "semantic";
}) {
  const width = Math.max(0, Math.min(100, value));
  return (
    <section className={`score-card score-${accent}`}>
      <div className="score-label">{title}</div>
      <div className="score-value">
        {formatNumber(value)}
        <span>%</span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${width}%` }} />
      </div>
    </section>
  );
}

function CompactScoreBar({ value }: { value?: number }) {
  if (typeof value !== "number") {
    return <span className="comparison-muted">Unavailable</span>;
  }

  const width = Math.max(0, Math.min(100, value));
  return (
    <div className="compact-score">
      <span>{formatNumber(value)}%</span>
      <div className="compact-track">
        <div className="compact-fill" style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}

function ModelComparisonPanel({ items }: { items: ModelComparisonItem[] }) {
  if (items.length === 0) return null;

  return (
    <section className="comparison-panel">
      <div className="comparison-header">
        <div>
          <div className="section-title">Model Comparison</div>
          <p>Same resume and job description, scored with the available encoder versions.</p>
        </div>
        <span>Skill scores stay constant; semantic scores show model behavior.</span>
      </div>

      <div className="comparison-table">
        <div className="comparison-row comparison-row-head">
          <span>Model</span>
          <span>Semantic</span>
          <span>Weighted Skills</span>
          <span>Overall</span>
          <span>Category</span>
        </div>
        {items.map((item) => (
          <div className="comparison-row" key={item.model_id}>
            <div className="comparison-model">
              <strong>{item.model_label}</strong>
              <small>{item.description}</small>
              <small>{item.model}</small>
            </div>
            <CompactScoreBar value={item.semantic_score} />
            <CompactScoreBar value={item.skill_match_score} />
            <CompactScoreBar value={item.overall_score} />
            <div className="comparison-category">
              {item.available ? item.match_category : item.error ?? "Unavailable"}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function SkillGroup({
  title,
  skills,
  tone
}: {
  title: string;
  skills: string[];
  tone: "match" | "missing" | "neutral";
}) {
  return (
    <section className="skill-card">
      <div className="section-title">{title}</div>
      <div className="chip-row">
        {skills.length === 0 ? (
          <span className="empty-state">No skills found</span>
        ) : (
          skills.map((skill) => (
            <span className={`chip chip-${tone}`} key={`${title}-${skill}`}>
              {skill}
            </span>
          ))
        )}
      </div>
    </section>
  );
}

function ListPanel({ title, items }: { title: string; items: string[] }) {
  return (
    <section className="analysis-panel">
      <div className="section-title">{title}</div>
      <ul>
        {items.length === 0 ? (
          <li>No items generated.</li>
        ) : (
          items.map((item) => <li key={`${title}-${item}`}>{item}</li>)
        )}
      </ul>
    </section>
  );
}

function EvidencePanel({ result }: { result: MatchResult }) {
  const visibleEvidence = result.skill_evidence.slice(0, 16);

  return (
    <section className="evidence-panel">
      <div className="evidence-header">
        <div>
          <div className="section-title">Skill Evidence</div>
          <p>
            Sentence-level classifier output for detected resume skills.
            {result.skill_evidence_model_available
              ? " Model validation is active."
              : " Dictionary-only fallback is active."}
          </p>
        </div>
        <span>{result.skill_evidence.length} mentions</span>
      </div>

      {visibleEvidence.length === 0 ? (
        <div className="empty-evidence">No resume skill evidence found.</div>
      ) : (
        <div className="evidence-list">
          {visibleEvidence.map((item, index) => (
            <article className={`evidence-item evidence-${item.label}`} key={`${item.skill}-${index}`}>
              <div className="evidence-meta">
                <strong>{item.skill}</strong>
                <span>{item.label}</span>
                <span>{formatNumber(item.confidence * 100)}%</span>
              </div>
              <p>{item.sentence}</p>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function ResultDashboard({ result }: { result: MatchResult }) {
  return (
    <section className="results-dashboard">
      <div className="summary-grid">
        <div className="score-grid">
          <ScoreCard title="Overall Suitability" value={result.overall_score} />
          <ScoreCard title="Semantic Similarity" value={result.semantic_score} accent="semantic" />
          <ScoreCard title="Weighted Skills" value={result.skill_match_score} />
          <ScoreCard title="Raw Skills" value={result.unweighted_skill_match_score} />
        </div>

        <section className="category-panel">
          <div>
            <div className="category-label">Result Category</div>
            <div className="category-value">{result.match_category}</div>
          </div>
          <p>Category combines semantic alignment with role-specific skill evidence.</p>
        </section>
      </div>

      <div className="skill-grid">
        <SkillGroup title="Matched Skills" skills={result.matched_skills} tone="match" />
        <SkillGroup title="Missing Skills" skills={result.missing_skills} tone="missing" />
        <SkillGroup title="Detected in Resume" skills={result.resume_skills} tone="neutral" />
        <SkillGroup title="Required by Job" skills={result.job_skills} tone="neutral" />
        <SkillGroup title="Negated Mentions" skills={result.negated_resume_skills} tone="missing" />
        <SkillGroup title="Unclear Mentions" skills={result.unclear_resume_skills} tone="neutral" />
      </div>

      <EvidencePanel result={result} />

      <div className="insight-grid">
        <section className="analysis-panel analysis-panel-large">
          <div className="section-title">HR Evaluation</div>
          <p>{result.hr_evaluation}</p>
        </section>
        <div className="side-insights">
          <ListPanel title="Interview Focus" items={result.interview_focus} />
          <ListPanel title="Candidate Suggestions" items={result.candidate_suggestions} />
        </div>
      </div>
    </section>
  );
}

function App() {
  const [resumeText, setResumeText] = useState("");
  const [jobText, setJobText] = useState("");
  const [result, setResult] = useState<MatchResult | null>(null);
  const [comparison, setComparison] = useState<ModelComparisonItem[]>([]);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingComparison, setLoadingComparison] = useState(false);
  const [loadingSample, setLoadingSample] = useState(false);
  const [uploadingTarget, setUploadingTarget] = useState<UploadTarget | null>(null);
  const [uploadInfo, setUploadInfo] = useState<Record<UploadTarget, UploadInfo>>({
    resume: null,
    job: null
  });
  const [error, setError] = useState("");

  const hasInput = useMemo(
    () => resumeText.trim().length > 0 && jobText.trim().length > 0,
    [resumeText, jobText]
  );
  const canAnalyze = useMemo(
    () => hasInput && !loading && !loadingComparison && !loadingSample && uploadingTarget === null,
    [hasInput, loading, loadingComparison, loadingSample, uploadingTarget]
  );
  const canCompare = useMemo(
    () => hasInput && !loading && !loadingComparison && !loadingSample && uploadingTarget === null,
    [hasInput, loading, loadingComparison, loadingSample, uploadingTarget]
  );

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((err: Error) => setError(`Backend is not reachable: ${err.message}`));
  }, []);

  async function loadExample() {
    setLoadingSample(true);
    setError("");
    try {
      const sample = await getSample();
      setResumeText(sample.resume_text);
      setJobText(sample.job_description_text);
      setResult(null);
      setComparison([]);
      setUploadInfo({ resume: null, job: null });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load sample data.");
    } finally {
      setLoadingSample(false);
    }
  }

  async function analyze() {
    if (!canAnalyze) return;
    setLoading(true);
    setError("");
    try {
      const analysis = await analyzeMatch({
        resume_text: resumeText,
        job_description_text: jobText
      });
      setResult(analysis);
      setComparison([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  async function compare() {
    if (!canCompare) return;
    setLoadingComparison(true);
    setError("");
    try {
      const payload = await compareModels({
        resume_text: resumeText,
        job_description_text: jobText
      });
      setComparison(payload.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Model comparison failed.");
    } finally {
      setLoadingComparison(false);
    }
  }

  async function uploadDocument(target: UploadTarget, file: File | undefined) {
    if (!file) return;

    setUploadingTarget(target);
    setError("");
    try {
      const extracted = await extractText(file);
      if (target === "resume") {
        setResumeText(extracted.extracted_text);
      } else {
        setJobText(extracted.extracted_text);
      }
      setUploadInfo((current) => ({ ...current, [target]: extracted }));
      setResult(null);
      setComparison([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not extract text from the uploaded file.");
    } finally {
      setUploadingTarget(null);
    }
  }

  return (
    <main className="page-shell">
      <header className="topbar">
        <div>
          <h1>Resume-Job Match Analysis</h1>
          <p>
            Explainable NLP dashboard for semantic resume-job matching, skill gap analysis,
            and HR screening support.
          </p>
        </div>
        <div className="badge-row">
          <span className="badge badge-primary">Fine-tuned MiniLM encoder</span>
          <span className="badge badge-neutral">Decision-support prototype</span>
          <span className="badge">{health?.model ?? "Backend pending"}</span>
        </div>
      </header>

      <section className="method-strip">
        <span className="method-label">Pipeline</span>
        {pipelineSteps.map((step) => (
          <span className="method-badge" key={step}>
            {step}
          </span>
        ))}
      </section>

      {error && <div className="error-banner">{error}</div>}

      <section className="input-grid">
        <div className="input-panel">
          <div className="input-heading">
            <span>Resume Content</span>
            <span>{resumeText.length.toLocaleString()} chars</span>
          </div>
          <div className="upload-strip">
            <label className="upload-button">
              {uploadingTarget === "resume" ? "Extracting..." : "Upload Resume"}
              <input
                type="file"
                accept=".pdf,.docx,.txt,.md"
                disabled={uploadingTarget !== null || loading || loadingComparison}
                onChange={(event) => {
                  void uploadDocument("resume", event.target.files?.[0]);
                  event.target.value = "";
                }}
              />
            </label>
            <span>
              {uploadInfo.resume
                ? `${uploadInfo.resume.file_name} - ${uploadInfo.resume.character_count.toLocaleString()} chars`
                : "PDF, DOCX, TXT, or MD"}
            </span>
          </div>
          <textarea
            value={resumeText}
            onChange={(event) => setResumeText(event.target.value)}
            spellCheck={false}
            placeholder="Paste the candidate resume text here..."
          />
        </div>

        <div className="input-panel">
          <div className="input-heading">
            <span>Job Description</span>
            <span>{jobText.length.toLocaleString()} chars</span>
          </div>
          <div className="upload-strip">
            <label className="upload-button">
              {uploadingTarget === "job" ? "Extracting..." : "Upload Job File"}
              <input
                type="file"
                accept=".pdf,.docx,.txt,.md"
                disabled={uploadingTarget !== null || loading || loadingComparison}
                onChange={(event) => {
                  void uploadDocument("job", event.target.files?.[0]);
                  event.target.value = "";
                }}
              />
            </label>
            <span>
              {uploadInfo.job
                ? `${uploadInfo.job.file_name} - ${uploadInfo.job.character_count.toLocaleString()} chars`
                : "PDF, DOCX, TXT, or MD"}
            </span>
          </div>
          <textarea
            value={jobText}
            onChange={(event) => setJobText(event.target.value)}
            spellCheck={false}
            placeholder="Paste the job description text here..."
          />
        </div>
      </section>

      <section className="action-row">
        <button
          className="button button-secondary"
          onClick={loadExample}
          disabled={loadingSample || loading || loadingComparison || uploadingTarget !== null}
        >
          {loadingSample ? "Loading..." : "Load Example"}
        </button>
        <button className="button button-primary" onClick={analyze} disabled={!canAnalyze}>
          {loading ? "Analyzing..." : "Analyze Match"}
        </button>
        <button className="button button-secondary" onClick={compare} disabled={!canCompare}>
          {loadingComparison ? "Comparing..." : "Compare Models"}
        </button>
      </section>

      <section className="result-heading">
        <h2>Analysis Result</h2>
        <span>Scores are decision-support signals, not automated hiring decisions.</span>
      </section>

      {result ? (
        <>
          <ResultDashboard result={result} />
          <ModelComparisonPanel items={comparison} />
        </>
      ) : comparison.length > 0 ? (
        <ModelComparisonPanel items={comparison} />
      ) : (
        <section className="empty-result">
          Load the example or paste your own resume and job description, then run the analysis.
        </section>
      )}
    </main>
  );
}

export default App;
