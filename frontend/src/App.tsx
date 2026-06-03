import { useEffect, useMemo, useState } from "react";
import { analyzeMatch, extractText, getHealth, getSample } from "./api";
import type { ExtractedDocumentPayload, HealthPayload, MatchResult } from "./types";

const pipelineSteps = [
  "Özel bi-encoder",
  "Doğrulama ile seçilen model",
  "Beceri çıkarımı",
  "Beceri kanıtı"
];

type UploadTarget = "resume" | "job";
type UploadInfo = ExtractedDocumentPayload | null;

function formatNumber(value: number): string {
  return value.toFixed(2);
}

function formatCharacterCount(value: number): string {
  return `${value.toLocaleString("tr-TR")} karakter`;
}

function translateFitLabel(label?: string | null): string {
  if (!label) return "";
  const labels: Record<string, string> = {
    "No Fit": "Uygun Değil",
    "Potential Fit": "Potansiyel Uygun",
    "Good Fit": "İyi Uygun"
  };
  return labels[label] ?? label;
}

function translateEvidenceLabel(label: string): string {
  const labels: Record<string, string> = {
    positive: "pozitif",
    negated: "olumsuz",
    unclear: "belirsiz"
  };
  return labels[label] ?? label;
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
          <span className="empty-state">Beceri bulunamadı</span>
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

function EvidencePanel({ result }: { result: MatchResult }) {
  const previewLimit = 16;
  const [showAllEvidence, setShowAllEvidence] = useState(false);
  const hasHiddenEvidence = result.skill_evidence.length > previewLimit;
  const visibleEvidence = showAllEvidence
    ? result.skill_evidence
    : result.skill_evidence.slice(0, previewLimit);

  return (
    <section className="evidence-panel">
      <div className="evidence-header">
        <div>
          <div className="section-title">Beceri Kanıtı</div>
          <p>
            CV içinde tespit edilen beceriler için cümle düzeyinde sınıflandırıcı çıktısı.
            {result.skill_evidence_model_available
              ? " Model doğrulaması aktif."
              : " Yalnızca sözlük tabanlı yedek akış aktif."}
          </p>
        </div>
        <div className="evidence-count">
          <strong>{result.skill_evidence.length.toLocaleString("tr-TR")} ifade</strong>
          {hasHiddenEvidence && !showAllEvidence && (
            <span>İlk {previewLimit.toLocaleString("tr-TR")} gösteriliyor</span>
          )}
        </div>
      </div>

      {visibleEvidence.length === 0 ? (
        <div className="empty-evidence">CV içinde beceri kanıtı bulunamadı.</div>
      ) : (
        <div className="evidence-list">
          {visibleEvidence.map((item, index) => (
            <article className={`evidence-item evidence-${item.label}`} key={`${item.skill}-${index}`}>
              <div className="evidence-meta">
                <strong>{item.skill}</strong>
                <span>{translateEvidenceLabel(item.label)}</span>
                <span>{formatNumber(item.confidence * 100)}%</span>
              </div>
              <p>{item.sentence}</p>
            </article>
          ))}
        </div>
      )}

      {hasHiddenEvidence && (
        <div className="evidence-actions">
          <button
            className="evidence-toggle"
            type="button"
            onClick={() => setShowAllEvidence((current) => !current)}
          >
            {showAllEvidence
              ? "Daha az göster"
              : `Tüm ${result.skill_evidence.length.toLocaleString("tr-TR")} ifadeyi göster`}
          </button>
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
          <ScoreCard title="Genel Uygunluk" value={result.overall_score} />
          <ScoreCard title="Anlamsal Benzerlik" value={result.semantic_score} accent="semantic" />
          <ScoreCard title="Ağırlıklı Beceriler" value={result.skill_match_score} />
          <ScoreCard title="Ham Beceri Skoru" value={result.unweighted_skill_match_score} />
        </div>

        <section className="category-panel">
          <div>
            <div className="category-label">Sonuç Kategorisi</div>
            <div className="category-value">{result.match_category}</div>
          </div>
          {result.predicted_fit_label && (
            <p>
              Model tahmini: <strong>{translateFitLabel(result.predicted_fit_label)}</strong>
              {typeof result.model_temperature === "number"
                ? `, sıcaklık ${formatNumber(result.model_temperature)}`
                : ""}
            </p>
          )}
          <p>Kategori, anlamsal uyumu role özgü beceri kanıtlarıyla birlikte değerlendirir.</p>
        </section>
      </div>

      <div className="skill-grid">
        <SkillGroup title="Eşleşen Beceriler" skills={result.matched_skills} tone="match" />
        <SkillGroup title="Eksik Beceriler" skills={result.missing_skills} tone="missing" />
        <SkillGroup title="CV'de Tespit Edilenler" skills={result.resume_skills} tone="neutral" />
        <SkillGroup title="İlanda İstenenler" skills={result.job_skills} tone="neutral" />
        <SkillGroup title="Olumsuz İfadeler" skills={result.negated_resume_skills} tone="missing" />
        <SkillGroup title="Belirsiz İfadeler" skills={result.unclear_resume_skills} tone="neutral" />
      </div>

      <EvidencePanel result={result} />
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
    () => hasInput && !loading && !loadingSample && uploadingTarget === null,
    [hasInput, loading, loadingSample, uploadingTarget]
  );

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((err: Error) => setError(`Backend'e ulaşılamıyor: ${err.message}`));
  }, []);

  async function loadExample() {
    setLoadingSample(true);
    setError("");
    try {
      const sample = await getSample();
      setResumeText(sample.resume_text);
      setJobText(sample.job_description_text);
      setResult(null);
      setUploadInfo({ resume: null, job: null });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Örnek veri yüklenemedi.");
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
      setError(err instanceof Error ? err.message : "Analiz başarısız oldu.");
    } finally {
      setLoading(false);
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
    } catch (err) {
      setError(err instanceof Error ? err.message : "Yüklenen dosyadan metin çıkarılamadı.");
    } finally {
      setUploadingTarget(null);
    }
  }

  return (
    <main className="page-shell">
      <header className="topbar">
        <div>
          <h1>CV-İş İlanı Eşleşme Analizi</h1>
          <p>
            Anlamsal CV-iş ilanı eşleştirme, beceri açığı analizi ve kanıt tabanlı
            uygunluk skoru için açıklanabilir NLP paneli.
          </p>
        </div>
        <div className="badge-row">
          <span className="badge badge-primary">Özel MiniLM bi-encoder</span>
          <span className="badge badge-neutral">Karar destek prototipi</span>
          <span className="badge">{health?.model ?? "Backend bekleniyor"}</span>
        </div>
      </header>

      <section className="method-strip">
        <span className="method-label">Süreç</span>
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
            <span>CV İçeriği</span>
            <span>{formatCharacterCount(resumeText.length)}</span>
          </div>
          <div className="upload-strip">
            <label className="upload-button">
              {uploadingTarget === "resume" ? "Metin çıkarılıyor..." : "CV Yükle"}
              <input
                type="file"
                accept=".pdf,.docx,.txt,.md"
                disabled={uploadingTarget !== null || loading}
                onChange={(event) => {
                  void uploadDocument("resume", event.target.files?.[0]);
                  event.target.value = "";
                }}
              />
            </label>
            <span>
              {uploadInfo.resume
                ? `${uploadInfo.resume.file_name} - ${formatCharacterCount(uploadInfo.resume.character_count)}`
                : "PDF, DOCX, TXT veya MD"}
            </span>
          </div>
          <textarea
            value={resumeText}
            onChange={(event) => setResumeText(event.target.value)}
            spellCheck={false}
            placeholder="Adayın CV metnini buraya yapıştırın..."
          />
        </div>

        <div className="input-panel">
          <div className="input-heading">
            <span>İş İlanı</span>
            <span>{formatCharacterCount(jobText.length)}</span>
          </div>
          <div className="upload-strip">
            <label className="upload-button">
              {uploadingTarget === "job" ? "Metin çıkarılıyor..." : "İlan Dosyası Yükle"}
              <input
                type="file"
                accept=".pdf,.docx,.txt,.md"
                disabled={uploadingTarget !== null || loading}
                onChange={(event) => {
                  void uploadDocument("job", event.target.files?.[0]);
                  event.target.value = "";
                }}
              />
            </label>
            <span>
              {uploadInfo.job
                ? `${uploadInfo.job.file_name} - ${formatCharacterCount(uploadInfo.job.character_count)}`
                : "PDF, DOCX, TXT veya MD"}
            </span>
          </div>
          <textarea
            value={jobText}
            onChange={(event) => setJobText(event.target.value)}
            spellCheck={false}
            placeholder="İş ilanı metnini buraya yapıştırın..."
          />
        </div>
      </section>

      <section className="action-row">
        <button
          className="button button-secondary"
          onClick={loadExample}
          disabled={loadingSample || loading || uploadingTarget !== null}
        >
          {loadingSample ? "Yükleniyor..." : "Örnek Yükle"}
        </button>
        <button className="button button-primary" onClick={analyze} disabled={!canAnalyze}>
          {loading ? "Analiz ediliyor..." : "Analiz Et"}
        </button>
      </section>

      <section className="result-heading">
        <h2>Analiz Sonucu</h2>
        <span>Skorlar karar destek sinyalidir; otomatik işe alım kararı değildir.</span>
      </section>

      {result ? (
        <ResultDashboard result={result} />
      ) : (
        <section className="empty-result">
          Örneği yükleyin veya kendi CV ve iş ilanı metninizi yapıştırıp analizi çalıştırın.
        </section>
      )}
    </main>
  );
}

export default App;
