import { useEffect, useMemo, useState } from "react";
import { analyzeMatch, getHealth, getSample } from "./api";
import type { HealthPayload, MatchResult } from "./types";

const pipelineSteps = [
  "Encoder embeddings",
  "Cosine similarity",
  "Skill extraction",
  "Explainable HR notes"
];

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
      </div>

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
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingSample, setLoadingSample] = useState(false);
  const [error, setError] = useState("");

  const canAnalyze = useMemo(
    () => resumeText.trim().length > 0 && jobText.trim().length > 0 && !loading,
    [resumeText, jobText, loading]
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
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setLoading(false);
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
          <textarea
            value={jobText}
            onChange={(event) => setJobText(event.target.value)}
            spellCheck={false}
            placeholder="Paste the job description text here..."
          />
        </div>
      </section>

      <section className="action-row">
        <button className="button button-secondary" onClick={loadExample} disabled={loadingSample || loading}>
          {loadingSample ? "Loading..." : "Load Example"}
        </button>
        <button className="button button-primary" onClick={analyze} disabled={!canAnalyze}>
          {loading ? "Analyzing..." : "Analyze Match"}
        </button>
      </section>

      <section className="result-heading">
        <h2>Analysis Result</h2>
        <span>Scores are decision-support signals, not automated hiring decisions.</span>
      </section>

      {result ? (
        <ResultDashboard result={result} />
      ) : (
        <section className="empty-result">
          Load the example or paste your own resume and job description, then run the analysis.
        </section>
      )}
    </main>
  );
}

export default App;
