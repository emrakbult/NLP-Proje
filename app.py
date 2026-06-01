from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from src.matcher import match_resume_to_job
from src.similarity import DEFAULT_MODEL_NAME, load_model
from src.skill_extractor import load_skills


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_RESUME_PATH = PROJECT_ROOT / "data" / "examples" / "sample_resume.txt"
SAMPLE_JOB_PATH = PROJECT_ROOT / "data" / "examples" / "sample_job.txt"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@st.cache_resource(show_spinner="Fine-tuned encoder yükleniyor...")
def get_model():
    return load_model(DEFAULT_MODEL_NAME)


@st.cache_resource
def get_skill_dictionary():
    return load_skills()


def compact_model_name(model_name: str) -> str:
    path = Path(model_name)
    return path.name if path.exists() else model_name


def score_color(score: float) -> str:
    if score >= 75:
        return "#15803d"
    if score >= 45:
        return "#b45309"
    return "#b91c1c"


def render_badge(text: str, class_name: str = "badge") -> str:
    return f"<span class='{class_name}'>{escape(text)}</span>"


def render_score_card(title: str, score: float, accent: str | None = None) -> None:
    st.markdown(score_card_html(title, score, accent), unsafe_allow_html=True)


def render_chip_group(title: str, skills: list[str], tone: str = "neutral") -> None:
    st.markdown(chip_group_html(title, skills, tone), unsafe_allow_html=True)


def score_card_html(title: str, score: float, accent: str | None = None) -> str:
    accent = accent or score_color(score)
    clamped_score = max(0.0, min(100.0, score))
    return f"""
    <div class="score-card" style="--accent:{accent};">
        <div class="score-label">{escape(title)}</div>
        <div class="score-main">
            <div class="score-value">{score:.2f}<span>%</span></div>
        </div>
        <div class="progress-track">
            <div class="progress-fill" style="width:{clamped_score:.2f}%;"></div>
        </div>
    </div>
    """


def chip_group_html(title: str, skills: list[str], tone: str = "neutral") -> str:
    chips = "".join(
        f"<span class='chip chip-{tone}'>{escape(skill)}</span>"
        for skill in skills
    )
    if not chips:
        chips = "<span class='empty-state'>Beceri bulunamadı</span>"

    return f"""
    <section class="evidence-section">
        <div class="section-title">{escape(title)}</div>
        <div class="chip-row">{chips}</div>
    </section>
    """


def render_text_panel(title: str, body: str) -> None:
    st.markdown(
        f"""
        <section class="analysis-panel analysis-panel-large">
            <div class="section-title">{escape(title)}</div>
            <p>{escape(body)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_list_panel(title: str, items: list[str]) -> None:
    st.markdown(list_panel_html(title, items), unsafe_allow_html=True)


def list_panel_html(title: str, items: list[str]) -> str:
    if items:
        list_items = "".join(f"<li>{escape(item)}</li>" for item in items)
    else:
        list_items = "<li>Öğe oluşturulmadı.</li>"

    return f"""
    <section class="analysis-panel">
        <div class="section-title">{escape(title)}</div>
        <ul>{list_items}</ul>
    </section>
    """


def result_dashboard_html(result: dict[str, object]) -> str:
    score_cards = "".join(
        [
            score_card_html("Genel Uygunluk", float(result["overall_score"])),
            score_card_html("Semantik Benzerlik", float(result["semantic_score"]), "#0f766e"),
            score_card_html("Ağırlıklı Beceriler", float(result["skill_match_score"])),
            score_card_html("Ham Beceri Skoru", float(result["unweighted_skill_match_score"])),
        ]
    )
    skill_cards = "".join(
        [
            chip_group_html("Eşleşen Beceriler", result["matched_skills"], "match"),
            chip_group_html("Eksik Beceriler", result["missing_skills"], "missing"),
            chip_group_html("Özgeçmişte Tespit Edilenler", result["resume_skills"], "neutral"),
            chip_group_html("İlanda İstenenler", result["job_skills"], "neutral"),
        ]
    )
    right_panels = list_panel_html("Mülakat Odak Noktaları", result["interview_focus"])
    right_panels += list_panel_html("Aday İçin Öneriler", result["candidate_suggestions"])

    return f"""
    <section class="results-dashboard">
        <div class="summary-grid">
            <div class="score-grid">
                {score_cards}
            </div>
            <section class="category-panel">
                <div>
                    <div class="category-label">Sonuç Kategorisi</div>
                    <div class="category-value">{escape(str(result["match_category"]))}</div>
                </div>
                <div class="category-caption">
                    Kategori, semantik uyumu role özgü beceri kanıtıyla birlikte değerlendirir.
                </div>
            </section>
        </div>

        <div class="skill-grid">
            {skill_cards}
        </div>

        <div class="insight-grid">
            <section class="analysis-panel analysis-panel-large">
                <div class="section-title">İK Değerlendirmesi</div>
                <p>{escape(str(result["hr_evaluation"]))}</p>
            </section>
            <div class="side-insights">
                {right_panels}
            </div>
        </div>
    </section>
    """


def render_methodology_strip() -> None:
    steps = [
        "Encoder vektörleri",
        "Kosinüs benzerliği",
        "Beceri çıkarımı",
        "Açıklanabilir İK notları",
    ]
    step_html = "".join(render_badge(step, "method-badge") for step in steps)
    st.markdown(
        f"""
        <div class="method-strip">
            <span class="method-label">İş Akışı</span>
            {step_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Özgeçmiş-İş Uygunluk Analizi",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #f5f7fa;
        --surface: #ffffff;
        --surface-soft: #f8fafc;
        --border: #e2e8f0;
        --border-strong: #cbd5e1;
        --text: #111827;
        --muted: #5b6472;
        --primary: #2563eb;
        --semantic: #0f766e;
        --positive: #15803d;
        --warning: #b45309;
        --critical: #b91c1c;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1380px;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, p {
        letter-spacing: 0;
    }

    h1 {
        font-size: 1.55rem !important;
        line-height: 1.25 !important;
        margin-bottom: 0.15rem !important;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.85rem;
    }

    textarea {
        border-radius: 8px !important;
        border-color: var(--border) !important;
        font-size: 0.92rem !important;
        line-height: 1.45 !important;
    }

    .topbar {
        border: 1px solid var(--border);
        background: var(--surface);
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }

    .topbar-inner {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
    }

    .subtitle {
        color: var(--muted);
        font-size: 0.94rem;
        margin: 0;
        line-height: 1.4;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 8px;
        min-width: 320px;
    }

    .badge,
    .method-badge {
        display: inline-flex;
        align-items: center;
        min-height: 28px;
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 5px 9px;
        background: var(--surface-soft);
        color: #334155;
        font-size: 0.78rem;
        font-weight: 600;
        line-height: 1.2;
        white-space: nowrap;
    }

    .badge-primary {
        border-color: #bfdbfe;
        background: #eff6ff;
        color: #1d4ed8;
    }

    .badge-neutral {
        border-color: #cbd5e1;
        background: #f8fafc;
        color: #475569;
    }

    .method-strip {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--surface);
        padding: 10px 12px;
    }

    .method-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-right: 2px;
    }

    .input-heading {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        border: 1px solid var(--border);
        border-bottom: 0;
        border-radius: 8px 8px 0 0;
        background: var(--surface);
        padding: 11px 12px;
        margin-bottom: -0.4rem;
    }

    .input-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: var(--text);
    }

    .input-meta {
        color: var(--muted);
        font-size: 0.78rem;
        white-space: nowrap;
    }

    .stTextArea textarea {
        border-radius: 0 0 8px 8px !important;
        min-height: 360px;
        background: #ffffff;
    }

    .action-row {
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--surface);
        padding: 12px;
    }

    div[data-testid="stButton"] button {
        border-radius: 4px;
        min-height: 42px;
        font-weight: 700;
    }

    .result-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .result-header h2 {
        font-size: 1.1rem !important;
        margin: 0 !important;
    }

    .result-note {
        color: var(--muted);
        font-size: 0.83rem;
    }

    .results-dashboard {
        display: grid;
        gap: 12px;
    }

    .summary-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 360px;
        gap: 12px;
        align-items: stretch;
    }

    .score-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
    }

    .score-card,
    .category-panel,
    .evidence-section,
    .analysis-panel,
    .empty-result {
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--surface);
    }

    .score-card {
        min-height: 108px;
        padding: 14px 16px;
        border-left: 4px solid var(--accent);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .score-label,
    .section-title,
    .category-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .score-value {
        color: var(--text);
        font-size: 1.62rem;
        line-height: 1.1;
        font-weight: 800;
    }

    .score-value span {
        color: var(--muted);
        font-size: 0.9rem;
        margin-left: 2px;
        font-weight: 700;
    }

    .progress-track {
        height: 7px;
        margin-top: 12px;
        background: #e5e7eb;
        border-radius: 999px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: var(--accent);
        border-radius: 999px;
    }

    .category-panel {
        padding: 16px;
        border-left: 4px solid var(--warning);
        background: #fffbeb;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 108px;
    }

    .category-value {
        margin-top: 6px;
        color: #78350f;
        font-size: 1.05rem;
        font-weight: 800;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }

    .category-caption {
        margin-top: 14px;
        color: #92400e;
        font-size: 0.82rem;
        line-height: 1.35;
    }

    .skill-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
    }

    .evidence-section {
        padding: 12px 14px;
        min-height: 92px;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        margin-top: 9px;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        border-radius: 4px;
        border: 1px solid var(--border);
        padding: 5px 8px;
        font-size: 0.8rem;
        line-height: 1.2;
        font-weight: 650;
        background: var(--surface-soft);
        color: #475569;
    }

    .chip-match {
        border-color: #bbf7d0;
        background: #f0fdf4;
        color: #166534;
    }

    .chip-missing {
        border-color: #fed7aa;
        background: #fff7ed;
        color: #9a3412;
    }

    .chip-neutral {
        border-color: #dbe2ea;
        background: #f8fafc;
        color: #475569;
    }

    .empty-state {
        color: var(--muted);
        font-size: 0.86rem;
    }

    .insight-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.08fr) minmax(360px, 0.92fr);
        gap: 12px;
        align-items: start;
    }

    .side-insights {
        display: grid;
        gap: 12px;
    }

    .analysis-panel {
        padding: 16px 18px;
        min-height: 0;
    }

    .analysis-panel-large {
        min-height: 220px;
    }

    .analysis-panel p,
    .analysis-panel li {
        color: #243041;
        font-size: 0.93rem;
        line-height: 1.55;
    }

    .analysis-panel p {
        margin: 12px 0 0;
    }

    .analysis-panel ul {
        margin: 12px 0 0;
        padding-left: 18px;
    }

    .analysis-panel li + li {
        margin-top: 8px;
    }

    .empty-result {
        padding: 28px 20px;
        color: var(--muted);
        text-align: center;
    }

    @media (max-width: 900px) {
        .topbar-inner {
            flex-direction: column;
        }

        .badge-row {
            justify-content: flex-start;
            min-width: 0;
        }

        .summary-grid,
        .insight-grid {
            grid-template-columns: 1fr;
        }

        .score-grid,
        .skill-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 620px) {
        .score-grid,
        .skill-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "resume_text" not in st.session_state:
    st.session_state.resume_text = read_text(SAMPLE_RESUME_PATH)
if "job_text" not in st.session_state:
    st.session_state.job_text = read_text(SAMPLE_JOB_PATH)
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

model_badge = render_badge("İnce ayarlı MiniLM encoder", "badge badge-primary")
prototype_badge = render_badge("Karar destek prototipi", "badge badge-neutral")
model_name_badge = render_badge(compact_model_name(DEFAULT_MODEL_NAME), "badge")

st.markdown(
    f"""
    <header class="topbar">
        <div class="topbar-inner">
            <div>
                <h1>Özgeçmiş-İş Uygunluk Analizi</h1>
                <p class="subtitle">
                    Semantik özgeçmiş-iş eşleştirme, beceri açığı analizi ve İK ön eleme desteği için
                    açıklanabilir NLP paneli.
                </p>
            </div>
            <div class="badge-row">
                {model_badge}
                {prototype_badge}
                {model_name_badge}
            </div>
        </div>
    </header>
    """,
    unsafe_allow_html=True,
)

render_methodology_strip()

input_left, input_right = st.columns(2)
with input_left:
    resume_text = st.session_state.resume_text
    st.markdown(
        f"""
        <div class="input-heading">
            <span class="input-title">Özgeçmiş İçeriği</span>
            <span class="input-meta">{len(resume_text):,} karakter</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    resume_text = st.text_area(
        "Özgeçmiş İçeriği",
        key="resume_text",
        height=360,
        label_visibility="collapsed",
    )

with input_right:
    job_text = st.session_state.job_text
    st.markdown(
        f"""
        <div class="input-heading">
            <span class="input-title">İş İlanı Metni</span>
            <span class="input-meta">{len(job_text):,} karakter</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    job_text = st.text_area(
        "İş İlanı Metni",
        key="job_text",
        height=360,
        label_visibility="collapsed",
    )

st.markdown("<div class='action-row'>", unsafe_allow_html=True)
action_left, action_right = st.columns([0.28, 0.72])
with action_left:
    load_example = st.button("Örneği Yükle", use_container_width=True)
with action_right:
    analyze = st.button("Uyumluluğu Analiz Et", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if load_example:
    st.session_state.resume_text = read_text(SAMPLE_RESUME_PATH)
    st.session_state.job_text = read_text(SAMPLE_JOB_PATH)
    st.session_state.analysis_result = None
    st.rerun()

if analyze:
    if not resume_text.strip() or not job_text.strip():
        st.error("Özgeçmiş ve iş ilanı metni boş olamaz.")
    else:
        with st.spinner("Özgeçmiş-iş uyumu fine-tuned encoder ile analiz ediliyor..."):
            st.session_state.analysis_result = match_resume_to_job(
                resume_text=resume_text,
                job_description_text=job_text,
                model=get_model(),
                skill_dictionary=get_skill_dictionary(),
            )

result = st.session_state.analysis_result

st.markdown(
    """
    <div class="result-header">
        <h2>Analiz Sonucu</h2>
        <span class="result-note">Skorlar karar destek sinyalidir; otomatik işe alım kararı değildir.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if result:
    st.markdown(result_dashboard_html(result), unsafe_allow_html=True)
else:
    st.markdown(
        """
        <section class="empty-result">
            Örneği yükleyin veya kendi özgeçmiş ve iş ilanı metninizi yapıştırıp analizi çalıştırın.
        </section>
        """,
        unsafe_allow_html=True,
    )
