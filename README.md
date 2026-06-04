# HR-Oriented Explainable Resume-Job Matching System

This project is an explainable NLP-based resume and job description matching system for human resources use cases.

It compares a candidate resume with a job description and returns:

- Overall suitability score
- Neural fit prediction: `No Fit`, `Potential Fit`, or `Good Fit`
- Cosine-based semantic similarity score
- Weighted skill match score
- Matched skills
- Missing or unclear required skills
- Negated and unclear resume skill evidence
- PDF/DOCX/TXT/Markdown upload for resume and job description

The system is a decision-support prototype. It is not intended to make automated hiring decisions.

## How To Start The Project

Use two terminals: one for the FastAPI backend and one for the React Vite frontend.

### 1. Pull Git LFS Model Files

Model weights are part of the project and are stored with Git LFS. After cloning the repository, run:

```powershell
git lfs install
git lfs pull
```

The application expects these local model folders to exist:

```text
models/base-minilm/
models/resume-job-biencoder-optimal/
models/skill-evidence-minilm-classifier/
```

### 2. Create And Activate The Python Environment

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

### 3. Start The Backend

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

### 4. Start The Frontend

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

## Runtime Flow

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
Custom ResumeJobBiEncoder
        |
        +--> Cosine-based semantic score
        |
        +--> 3-class fit prediction
        |
        v
Skill extraction + skill evidence classifier
        |
        v
Weighted skill match score
        |
        v
Explainable score, category, skill gaps, and evidence
```

## Model Approach

The final runtime model is a custom encoder-only bi-encoder:

```text
models/resume-job-biencoder-optimal/
```

It uses local pretrained MiniLM weights from:

```text
models/base-minilm/
```

The architecture uses:

- Shared MiniLM encoder for resume and job text
- Mean pooling over token embeddings
- Projection head: `384 -> 256 -> 128`
- L2-normalized projected embeddings
- Cosine similarity with a learnable temperature parameter
- Pair classification head over `[resume, job, abs diff, elementwise product]`

The optimal model was selected from a 25-epoch training run by lowest validation total loss. The selected checkpoint is epoch 22.

The project also uses a second encoder-only classifier for sentence-level skill evidence validation:

```text
models/skill-evidence-minilm-classifier/
```

This classifier checks whether a resume sentence provides positive evidence for a skill, negates the skill, or mentions it unclearly.

## Backend API

The FastAPI backend provides:

```text
GET  /health
GET  /sample
POST /analyze
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

The response includes scores, neural fit prediction, skill lists, match category, and skill evidence. It does not generate free-form HR evaluation text.

Use `POST /extract-text` with multipart field `file` to extract text from `.pdf`, `.docx`, `.txt`, or `.md` files. The React UI uses this endpoint for both resume and job description uploads.

## Reports

Training and evaluation artifacts are stored under:

```text
reports/
reports/figures/
```

Important files:

- `reports/biencoder_training_config.json`
- `reports/biencoder_training_history.csv`
- `reports/biencoder_optimal_selection.json`
- `reports/biencoder_test_metrics.json`
- `reports/biencoder_test_predictions.csv`
- `reports/model_comparison_metrics.csv`
- `reports/figures/loss_total.png`
- `reports/figures/loss_cosine.png`
- `reports/figures/loss_classification.png`
- `reports/figures/validation_accuracy.png`
- `reports/figures/validation_correlations.png`
- `reports/figures/score_distribution_by_label.png`
- `reports/figures/confusion_matrix.png`

Detailed experiment interpretation is documented in `RESULTS.md`.

## Project Structure

```text
NLP-Proje/
|-- README.md
|-- LICENSE
|-- IDEA.md
|-- PLAN.md
|-- PROJECT_SUMMARY.md
|-- RESULTS.md
|-- requirements.txt
|-- api.py
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
|   |-- base-minilm/
|   |-- resume-job-biencoder-optimal/
|   `-- skill-evidence-minilm-classifier/
|-- reports/
|-- scripts/
|-- src/
|   |-- neural_matcher.py
|   |-- neural_training.py
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
- `src/neural_matcher.py`: custom bi-encoder architecture and runtime wrapper
- `src/neural_training.py`: dataset, loss, epoch metrics, and checkpoint selection helpers
- `src/similarity.py`: runtime model loading and cosine utilities
- `src/matcher.py`: final matching pipeline
- `src/skill_extractor.py`: skill detection and alias handling
- `src/skill_evidence.py`: sentence-level positive, negated, and unclear skill evidence classification
- `src/skill_weights.py`: role-specific skill weighting
- `src/recommender.py`: deterministic match category rules
- `scripts/train_biencoder.py`: custom bi-encoder training pipeline
- `scripts/evaluate_biencoder.py`: final test evaluation and model comparison report
- `scripts/plot_training_history.py`: training and evaluation plot generation

## Run Tests

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Notes

- Use `http://localhost:5173` for the main demo.
- Keep the backend running while using the frontend.
- Runtime uses the local optimal bi-encoder model.
- Model comparison belongs to `RESULTS.md` and `reports/`, not the UI.
- Demo upload PDFs and local logs are ignored and should not be committed.

## License And Attribution

This project code is released under the MIT License. See `LICENSE`.

External resources used by the project:

- Base encoder: `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face, licensed as Apache-2.0.
- Main dataset: `cnamuangtoun/resume-job-description-fit` from Hugging Face. Local CSV copies are included for reproducible coursework; use them according to the source dataset terms.
- Document extraction: Microsoft MarkItDown, licensed as MIT.

The system is an educational decision-support prototype. It should not be used as an automated hiring decision system.
