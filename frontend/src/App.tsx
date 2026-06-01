import { useEffect, useMemo, useState } from "react";
import { analyzeMatch, compareModels, extractText, getHealth, getSample } from "./api";
import type { ExtractedDocumentPayload, HealthPayload, MatchResult, ModelComparisonItem } from "./types";

type UploadTarget = "resume" | "job";
type UploadInfo = ExtractedDocumentPayload | null;
type ScoreTone = "good" | "warning" | "bad" | "semantic";

const workflowItems = ["Özgeçmiş", "İş ilanı", "Analiz", "Karşılaştırma"];

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

function scoreTone(score: number): ScoreTone {
  if (score >= 75) return "good";
  if (score >= 45) return "warning";
  return "bad";
}

function StatusBadge({ health }: { health: HealthPayload | null }) {
  const online = health?.status === "ok";

  return (
    <div className={`status-badge ${online ? "status-online" : "status-waiting"}`}>
      <span />
      <div>
        <strong>{online ? "Backend aktif" : "Backend bekleniyor"}</strong>
        <small>{health?.model ?? "Model yüklenmedi"}</small>
      </div>
    </div>
  );
}

function ScoreCard({
  title,
  value,
  tone = scoreTone(value)
}: {
  title: string;
  value: number;
  tone?: ScoreTone;
}) {
  const width = Math.max(0, Math.min(100, value));

  return (
    <article className={`score-card score-${tone}`}>
      <div className="score-card-top">
        <span>{title}</span>
        <strong>{formatNumber(value)}%</strong>
      </div>
      <div className="score-track" aria-hidden="true">
        <div className="score-fill" style={{ width: `${width}%` }} />
      </div>
    </article>
  );
}

function CompactScoreBar({ value }: { value?: number }) {
  if (typeof value !== "number") {
    return <span className="comparison-muted">Kullanılamıyor</span>;
  }

  const width = Math.max(0, Math.min(100, value));

  return (
    <div className="compact-score">
      <strong>{formatNumber(value)}%</strong>
      <div className="compact-track" aria-hidden="true">
        <div className="compact-fill" style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}

function TextInputPanel({
  title,
  value,
  uploadText,
  uploadBusyText,
  placeholder,
  uploadInfo,
  isUploading,
  disabled,
  onChange,
  onUpload
}: {
  title: string;
  value: string;
  uploadText: string;
  uploadBusyText: string;
  placeholder: string;
  uploadInfo: UploadInfo;
  isUploading: boolean;
  disabled: boolean;
  onChange: (value: string) => void;
  onUpload: (file: File | undefined) => void;
}) {
  const fileSummary = uploadInfo
    ? `${uploadInfo.file_name} · ${uploadInfo.character_count.toLocaleString()} karakter`
    : "PDF, DOCX, TXT veya MD";

  return (
    <section className="input-panel">
      <div className="input-header">
        <div>
          <span>{title}</span>
          <strong>{value.length.toLocaleString()} karakter</strong>
        </div>
        <label className={`upload-button ${disabled ? "upload-disabled" : ""}`}>
          {isUploading ? uploadBusyText : uploadText}
          <input
            type="file"
            accept=".pdf,.docx,.txt,.md"
            disabled={disabled}
            onChange={(event) => {
              onUpload(event.target.files?.[0]);
              event.target.value = "";
            }}
          />
        </label>
      </div>
      <div className="file-summary">{fileSummary}</div>
      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        spellCheck={false}
        placeholder={placeholder}
      />
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
    <section className="skill-panel">
      <div className="panel-title">{title}</div>
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
    <section className="text-panel">
      <div className="panel-title">{title}</div>
      <ul>
        {items.length === 0 ? <li>Öğe oluşturulmadı.</li> : items.map((item) => <li key={`${title}-${item}`}>{item}</li>)}
      </ul>
    </section>
  );
}

function EvidencePanel({ result }: { result: MatchResult }) {
  const visibleEvidence = result.skill_evidence.slice(0, 12);

  return (
    <section className="evidence-section">
      <div className="section-heading">
        <div>
          <span>Beceri Kanıtı</span>
          <strong>{result.skill_evidence.length} bahsetme</strong>
        </div>
        <p>
          {result.skill_evidence_model_available
            ? "Cümle düzeyinde model doğrulama aktif."
            : "Sözlük tabanlı yedek yöntem aktif."}
        </p>
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
      <div className="result-overview">
        <article className="result-main">
          <span>Genel Uygunluk</span>
          <div>
            <strong>{formatNumber(result.overall_score)}%</strong>
            <p>{result.match_category}</p>
          </div>
        </article>

        <div className="score-grid">
          <ScoreCard title="Semantik Benzerlik" value={result.semantic_score} tone="semantic" />
          <ScoreCard title="Ağırlıklı Beceriler" value={result.skill_match_score} />
          <ScoreCard title="Ham Beceri Skoru" value={result.unweighted_skill_match_score} />
        </div>
      </div>

      <div className="skill-grid">
        <SkillGroup title="Eşleşen Beceriler" skills={result.matched_skills} tone="match" />
        <SkillGroup title="Eksik Beceriler" skills={result.missing_skills} tone="missing" />
        <SkillGroup title="Özgeçmişte Tespit Edilenler" skills={result.resume_skills} tone="neutral" />
        <SkillGroup title="İlanda İstenenler" skills={result.job_skills} tone="neutral" />
        <SkillGroup title="Olumsuz İfadeler" skills={result.negated_resume_skills} tone="missing" />
        <SkillGroup title="Belirsiz İfadeler" skills={result.unclear_resume_skills} tone="neutral" />
      </div>

      <div className="insight-grid">
        <section className="text-panel text-panel-large">
          <div className="panel-title">İK Değerlendirmesi</div>
          <p>{result.hr_evaluation}</p>
        </section>
        <div className="side-insights">
          <ListPanel title="Mülakat Odak Noktaları" items={result.interview_focus} />
          <ListPanel title="Aday İçin Öneriler" items={result.candidate_suggestions} />
        </div>
      </div>

      <EvidencePanel result={result} />
    </section>
  );
}

function ModelComparisonPanel({ items }: { items: ModelComparisonItem[] }) {
  if (items.length === 0) return null;

  return (
    <section className="comparison-panel">
      <div className="section-heading">
        <div>
          <span>Model Karşılaştırması</span>
          <strong>{items.length} model</strong>
        </div>
        <p>Beceri skorları sabit kalır; semantik skorlar model davranışını gösterir.</p>
      </div>

      <div className="comparison-table">
        <div className="comparison-row comparison-row-head">
          <span>Model</span>
          <span>Semantik</span>
          <span>Beceri</span>
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

  const hasInput = useMemo(() => resumeText.trim().length > 0 && jobText.trim().length > 0, [resumeText, jobText]);
  const busy = loading || loadingComparison || loadingSample || uploadingTarget !== null;
  const canAnalyze = hasInput && !busy;
  const canCompare = hasInput && !busy;

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
      <header className="app-header">
        <div className="brand-block">
          <span className="eyebrow">Açıklanabilir NLP Paneli</span>
          <h1>Özgeçmiş-İş Uygunluk Analizi</h1>
          <p>Semantik eşleştirme, beceri açığı analizi ve İK değerlendirmesi tek ekranda.</p>
        </div>
        <StatusBadge health={health} />
      </header>

      <section className="workflow-strip" aria-label="İş akışı">
        {workflowItems.map((item, index) => (
          <span key={item}>
            <strong>{index + 1}</strong>
            {item}
          </span>
        ))}
      </section>

      {error && <div className="error-banner">{error}</div>}

      <section className="workspace-grid">
        <TextInputPanel
          title="Özgeçmiş"
          value={resumeText}
          uploadText="Dosya Yükle"
          uploadBusyText="Çıkarılıyor..."
          placeholder="Adayın özgeçmiş metnini buraya yapıştırın."
          uploadInfo={uploadInfo.resume}
          isUploading={uploadingTarget === "resume"}
          disabled={busy}
          onChange={setResumeText}
          onUpload={(file) => void uploadDocument("resume", file)}
        />
        <TextInputPanel
          title="İş İlanı"
          value={jobText}
          uploadText="Dosya Yükle"
          uploadBusyText="Çıkarılıyor..."
          placeholder="İş ilanı metnini buraya yapıştırın."
          uploadInfo={uploadInfo.job}
          isUploading={uploadingTarget === "job"}
          disabled={busy}
          onChange={setJobText}
          onUpload={(file) => void uploadDocument("job", file)}
        />
      </section>

      <section className="action-bar">
        <button
          className="button button-secondary"
          onClick={loadExample}
          disabled={busy}
          type="button"
        >
          {loadingSample ? "Yükleniyor..." : "Örneği Yükle"}
        </button>
        <button className="button button-primary" onClick={analyze} disabled={!canAnalyze} type="button">
          {loading ? "Analiz ediliyor..." : "Uyumluluğu Analiz Et"}
        </button>
        <button className="button button-secondary" onClick={compare} disabled={!canCompare} type="button">
          {loadingComparison ? "Karşılaştırılıyor..." : "Modelleri Karşılaştır"}
        </button>
      </section>

      <section className="result-shell">
        <div className="result-heading">
          <div>
            <span>Analiz Sonucu</span>
            <h2>{result ? result.match_category : comparison.length > 0 ? "Model karşılaştırması hazır" : "Henüz analiz yok"}</h2>
          </div>
          <p>Skorlar karar destek sinyalidir; otomatik işe alım kararı değildir.</p>
        </div>

        {result ? (
          <>
            <ResultDashboard result={result} />
            <ModelComparisonPanel items={comparison} />
          </>
        ) : comparison.length > 0 ? (
          <ModelComparisonPanel items={comparison} />
        ) : (
          <div className="empty-result">Örnek metni yükleyin veya kendi metinlerinizle analizi başlatın.</div>
        )}
      </section>
    </main>
  );
}

export default App;
