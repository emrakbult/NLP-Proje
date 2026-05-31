# HR-Oriented Explainable Resume-Job Matching System

This project is an explainable NLP-based resume and job description matching system for human resources use cases.

It compares a candidate resume with a job description and returns:

- Overall suitability score
- Semantic similarity score
- Weighted skill match score
- Matched skills
- Missing or unclear required skills
- HR-oriented evaluation text
- Interview focus suggestions
- Candidate improvement suggestions
- Model comparison against previous encoder versions
- PDF/DOCX/TXT upload for resume and job description
- Sentence-level skill evidence validation for positive, negated, and unclear skill mentions

The system is a decision-support prototype. It is not intended to make automated hiring decisions.

## How To Start The Project

Use two terminals: one for the FastAPI backend and one for the React Vite frontend.

### 1. Create And Activate The Python Environment

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the virtual environment already exists, only activate it:

```powershell
.\.venv\Scripts\activate
```

### 2. Start The Backend

From the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --port 8000
```

The backend should run at:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

### 3. Start The Frontend

Open a second terminal and run:

```powershell
cd frontend
npm install
npm run dev
```

The React UI should run at:

```text
http://localhost:5173
```

Open this address in the browser and use the interface to analyze a resume and job description.

## Main Runtime Flow

```text
Uploaded files or pasted text
        |
        v
MarkItDown text extraction for PDF/DOCX
        |
        v
Resume text + job description text
        |
        v
Encoder-only Sentence Transformer
        |
        v
Cosine similarity score
        |
        v
Dictionary-based skill extraction
        |
        v
Skill evidence classifier for resume mentions
        |
        v
Weighted skill match score
        |
        v
Overall suitability score
        |
        v
Explainable HR result
```

## Model Approach

The project uses an encoder-only Sentence Transformer model for semantic matching. The local model is fine-tuned for the resume-job matching task, so the system is adapted to compare candidate profiles and job requirements in the same embedding space.

The project also uses a second encoder-only classifier for skill evidence validation. This classifier checks whether a resume sentence provides positive evidence for a skill, negates the skill, or mentions it unclearly.

Fine-tuning is part of the project methodology, but this README focuses on running the completed system. Detailed experiment results are documented in `RESULTS.md`.

The final score combines semantic similarity and weighted skill matching:

```text
overall_score =
  0.80 * semantic_similarity_score +
  0.20 * weighted_skill_match_score
```

The model gives more weight to technical and role-specific skills than broad general skills. This prevents generic skills such as communication or teamwork from making a weak candidate look too strong.

## Backend API

The FastAPI backend provides:

```text
GET  /health
GET  /sample
POST /analyze
POST /compare
POST /extract-text
```

Main endpoint:

```text
POST /analyze
```

Expected JSON body:

```json
{
  "resume_text": "Candidate resume text...",
  "job_description_text": "Job description text..."
}
```

The response includes the scores, skill lists, match category, HR evaluation, interview focus, and candidate suggestions.

Use `POST /compare` to score the same resume-job pair with the available model versions and compare semantic, skill, and overall scores.

Use `POST /extract-text` with multipart field `file` to extract text from `.pdf`, `.docx`, `.txt`, or `.md` files. The React UI uses this endpoint for both resume and job description uploads.

## Project Structure

```text
NLP-Proje/
|-- README.md
|-- IDEA.md
|-- PLAN.md
|-- PROJECT_SUMMARY.md
|-- RESULTS.md
|-- requirements.txt
|-- api.py
|-- app.py
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- skill_evidence/
|   |-- skills.json
|   `-- examples/
|-- frontend/
|   |-- package.json
|   |-- index.html
|   `-- src/
|       |-- App.tsx
|       |-- api.ts
|       |-- styles.css
|       `-- types.ts
|-- models/
|-- scripts/
|-- src/
|   |-- similarity.py
|   |-- skill_extractor.py
|   |-- skill_weights.py
|   |-- matcher.py
|   |-- skill_evidence.py
|   |-- recommender.py
|   `-- evaluation.py
`-- tests/
```

## Important Files

- `api.py`: FastAPI backend used by the React interface
- `frontend/src/App.tsx`: main React UI
- `src/similarity.py`: encoder-only model loading and cosine similarity
- `src/skill_extractor.py`: skill detection and alias handling
- `src/skill_evidence.py`: sentence-level positive, negated, and unclear skill evidence classification
- `src/skill_weights.py`: role-specific skill weighting
- `src/matcher.py`: final matching pipeline
- `src/recommender.py`: match category, HR explanation, interview focus, and candidate suggestions
- `data/skill_evidence/skill_evidence_dataset.csv`: manually curated skill evidence classifier dataset
- `data/skills.json`: skill dictionary used for explainability
- `models/`: local Sentence Transformer model files used by the system

## Run Tests

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Optional Streamlit Demo

The main UI is the React Vite app. A Streamlit version is still available as a secondary Python-only demo:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Notes

- Use `http://localhost:5173` for the main demo.
- Keep the backend running while using the frontend.
- The backend loads the local model once and reuses it for analysis requests.
- Detailed experiment results are documented in `RESULTS.md`.
