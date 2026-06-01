import { useEffect, useMemo, useState } from "react";
import { analyzeMatch, compareModels, extractText, getHealth, getSample } from "./api";
import type { ExtractedDocumentPayload, HealthPayload, MatchResult, ModelComparisonItem } from "./types";

const pipelineSteps = [
  "Encoder vektörleri",
  "Kosinüs benzerliği",
  "Beceri çıkarımı",
  "Açıklanabilir İK notları"
];

type UploadTarget = "resume" | "job";
type UploadInfo = ExtractedDocumentPayload | null;

function formatNumber(value: number): string {
  return value.toFixed(2);
}

function formatEvidenceLabel(label: string): string {
  const labels: Record<string, string> = {
    positive: "Pozitif",
    negated: "Olumsuz",
    unclear: "Belirsiz"
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

function CompactScoreBar({ value }: { value?: number }) {
  if (typeof value !== "number") {
    return <span className="comparison-muted">Kullanılamıyor</span>;
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
          <div className="section-title">Model Karşılaştırması</div>
          <p>Aynı özgeçmiş ve iş ilanı mevcut encoder sürümleriyle skorlanır.</p>
        </div>
        <span>Beceri skorları sabit kalır; semantik skorlar model davranışını gösterir.</span>
      </div>

      <div className="comparison-table">
        <div className="comparison-row comparison-row-head">
          <span>Model</span>
          <span>Semantik</span>
          <span>Ağırlıklı Beceriler</span>
          <span>Genel</span>
          <span>Kategori</span>
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
              {item.available ? item.match_category : item.error ?? "Kullanılamıyor"}
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

function ListPanel({ title, items }: { title: string; items: string[] }) {
  return (
    <section className="analysis-panel">
      <div className="section-title">{title}</div>
      <ul>
        {items.length === 0 ? (
          <li>Öğe oluşturulmadı.</li>
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
          <div className="section-title">Beceri Kanıtı</div>
          <p>
            Özgeçmişte tespit edilen beceriler için cümle düzeyinde sınıflandırıcı çıktısı.
            {result.skill_evidence_model_available
              ? " Model doğrulama aktif durumda."
              : " Yalnızca sözlük tabanlı yedek yöntem aktif."}
          </p>
        </div>
        <span>{result.skill_evidence.length} bahsetme</span>
      </div>

      {visibleEvidence.length === 0 ? (
        <div className="empty-evidence">Özgeçmişte beceri kanıtı bulunamadı.</div>
      ) : (
        <div className="evidence-list">
          {visibleEvidence.map((item, index) => (
            <article className={`evidence-item evidence-${item.label}`} key={`${item.skill}-${index}`}>
              <div className="evidence-meta">
                <strong>{item.skill}</strong>
                <span>{formatEvidenceLabel(item.label)}</span>
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
          <ScoreCard title="Genel Uygunluk" value={result.overall_score} />
          <ScoreCard title="Semantik Benzerlik" value={result.semantic_score} accent="semantic" />
          <ScoreCard title="Ağırlıklı Beceriler" value={result.skill_match_score} />
          <ScoreCard title="Ham Beceri Skoru" value={result.unweighted_skill_match_score} />
        </div>

        <section className="category-panel">
          <div>
            <div className="category-label">Sonuç Kategorisi</div>
            <div className="category-value">{result.match_category}</div>
          </div>
          <p>Kategori, semantik uyumu role özgü beceri kanıtıyla birlikte değerlendirir.</p>
        </section>
      </div>

      <div className="skill-grid">
        <SkillGroup title="Eşleşen Beceriler" skills={result.matched_skills} tone="match" />
        <SkillGroup title="Eksik Beceriler" skills={result.missing_skills} tone="missing" />
        <SkillGroup title="Özgeçmişte Tespit Edilenler" skills={result.resume_skills} tone="neutral" />
        <SkillGroup title="İlanda İstenenler" skills={result.job_skills} tone="neutral" />
        <SkillGroup title="Olumsuz İfadeler" skills={result.negated_resume_skills} tone="missing" />
        <SkillGroup title="Belirsiz İfadeler" skills={result.unclear_resume_skills} tone="neutral" />
      </div>

      <EvidencePanel result={result} />

      <div className="insight-grid">
        <section className="analysis-panel analysis-panel-large">
          <div className="section-title">İK Değerlendirmesi</div>
          <p>{result.hr_evaluation}</p>
        </section>
        <div className="side-insights">
          <ListPanel title="Mülakat Odak Noktaları" items={result.interview_focus} />
          <ListPanel title="Aday İçin Öneriler" items={result.candidate_suggestions} />
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
      setComparison([]);
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
      setComparison([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analiz başarısız oldu.");
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
      setError(err instanceof Error ? err.message : "Model karşılaştırması başarısız oldu.");
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
      setError(err instanceof Error ? err.message : "Yüklenen dosyadan metin çıkarılamadı.");
    } finally {
      setUploadingTarget(null);
    }
  }

  return (
    <main className="page-shell">
      <header className="topbar">
        <div>
          <h1>Özgeçmiş-İş Uygunluk Analizi</h1>
          <p>
            Semantik özgeçmiş-iş eşleştirme, beceri açığı analizi ve İK ön eleme desteği için
            açıklanabilir NLP paneli.
          </p>
        </div>
        <div className="badge-row">
          <span className="badge badge-primary">İnce ayarlı MiniLM encoder</span>
          <span className="badge badge-neutral">Karar destek prototipi</span>
          <span className="badge">{health?.model ?? "Backend bekleniyor"}</span>
        </div>
      </header>

      <section className="method-strip">
        <span className="method-label">İş Akışı</span>
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
            <span>Özgeçmiş İçeriği</span>
            <span>{resumeText.length.toLocaleString()} karakter</span>
          </div>
          <div className="upload-strip">
            <label className="upload-button">
              {uploadingTarget === "resume" ? "Çıkarılıyor..." : "Özgeçmiş Yükle"}
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
                ? `${uploadInfo.resume.file_name} - ${uploadInfo.resume.character_count.toLocaleString()} karakter`
                : "PDF, DOCX, TXT veya MD"}
            </span>
          </div>
          <textarea
            value={resumeText}
            onChange={(event) => setResumeText(event.target.value)}
            spellCheck={false}
            placeholder="Adayın özgeçmiş metnini buraya yapıştırın..."
          />
        </div>

        <div className="input-panel">
          <div className="input-heading">
            <span>İş İlanı Metni</span>
            <span>{jobText.length.toLocaleString()} karakter</span>
          </div>
          <div className="upload-strip">
            <label className="upload-button">
              {uploadingTarget === "job" ? "Çıkarılıyor..." : "İş Dosyası Yükle"}
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
                ? `${uploadInfo.job.file_name} - ${uploadInfo.job.character_count.toLocaleString()} karakter`
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
          disabled={loadingSample || loading || loadingComparison || uploadingTarget !== null}
        >
          {loadingSample ? "Yükleniyor..." : "Örneği Yükle"}
        </button>
        <button className="button button-primary" onClick={analyze} disabled={!canAnalyze}>
          {loading ? "Analiz ediliyor..." : "Uyumluluğu Analiz Et"}
        </button>
        <button className="button button-secondary" onClick={compare} disabled={!canCompare}>
          {loadingComparison ? "Karşılaştırılıyor..." : "Modelleri Karşılaştır"}
        </button>
      </section>

      <section className="result-heading">
        <h2>Analiz Sonucu</h2>
        <span>Skorlar karar destek sinyalidir; otomatik işe alım kararı değildir.</span>
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
          Örneği yükleyin veya kendi özgeçmiş ve iş ilanı metninizi yapıştırıp analizi çalıştırın.
        </section>
      )}
    </main>
  );
}

export default App;
