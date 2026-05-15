# HR-Oriented Explainable Resume-Job Matching System

This project is an explainable Natural Language Processing system for human resources use cases. It compares a candidate resume with a job description and returns an interpretable suitability result.

The system is not intended to return only a raw similarity score. It also explains the result by showing matched skills, missing skills, and HR-oriented evaluation notes.

## Core Idea

Given:

- A resume or CV text
- A job description text

The system should produce:

- Overall suitability score
- Semantic similarity score
- Skill match score
- Matched skills
- Missing or unclear skills
- HR evaluation summary
- Technical interview focus suggestions

## Main NLP Approach

The project uses an encoder-only Sentence Transformer model for semantic similarity.

By default, the code first uses the current final model when it exists:

```text
models/resume-job-minilm-finetuned-2epoch/
```

If that model is not available, it falls back to the previous 1-epoch fine-tuned model:

```text
models/resume-job-minilm-finetuned/
```

If no local fine-tuned model is available, it falls back to:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Basic flow:

```text
Resume Text -------> Encoder-only Model -------> Resume Embedding
                                                    |
                                                    | Cosine Similarity
                                                    |
Job Text ----------> Encoder-only Model -------> Job Embedding
```

Skill-level explainability is handled with dictionary-based skill extraction.

## Dataset

The main dataset is:

```text
cnamuangtoun/resume-job-description-fit
```

Local dataset path:

```text
data/raw/cnamuangtoun_resume_job_description_fit/
```

Files:

```text
train.csv
test.csv
```

Columns:

- `resume_text`
- `job_description_text`
- `label`

Label classes:

- `Good Fit`
- `Potential Fit`
- `No Fit`

Current local dataset summary:

```text
train.csv: 6241 rows
test.csv: 1759 rows
total: 8000 rows
```

## Project Structure

```text
NLP-Proje/
|-- IDEA.md
|-- PLAN.md
|-- PROJECT_SUMMARY.md
|-- RESULTS.md
|-- README.md
|-- requirements.txt
|-- api.py
|-- app.py
|-- data/
|   |-- raw/
|   |   `-- cnamuangtoun_resume_job_description_fit/
|   |       |-- train.csv
|   |       `-- test.csv
|   |-- processed/
|   |-- skills.json
|   `-- examples/
|       |-- sample_resume.txt
|       `-- sample_job.txt
|-- scripts/
|   |-- inspect_dataset.py
|   |-- run_sample_matching.py
|   |-- analyze_errors.py
|   |-- evaluate_matching.py
|   |-- evaluate_ranking.py
|   `-- train_sentence_transformer.py
|-- src/
|   |-- __init__.py
|   |-- data_loader.py
|   |-- evaluation.py
|   |-- fine_tuning.py
|   |-- preprocessing.py
|   |-- skill_extractor.py
|   |-- skill_weights.py
|   |-- similarity.py
|   |-- recommender.py
|   `-- matcher.py
|-- tests/
    |-- conftest.py
    |-- test_evaluation.py
    |-- test_recommender.py
    |-- test_skill_extractor.py
    |-- test_similarity.py
    `-- test_matcher.py
`-- frontend/
    |-- package.json
    |-- index.html
    `-- src/
        |-- App.tsx
        |-- api.ts
        |-- styles.css
        `-- types.ts
```

## Setup

The project uses a local virtual environment at `.venv`.

Create it and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For GPU fine-tuning on Windows with an NVIDIA GPU, install the CUDA PyTorch wheel inside the same venv:

```bash
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

Verify that PyTorch can see the GPU:

```bash
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Or run commands directly through the venv Python:

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

## Dataset Inspection

Run the dataset inspection script:

```bash
.\.venv\Scripts\python.exe scripts/inspect_dataset.py
```

Expected output includes:

- Row count for each split
- Column names
- Label distribution
- A short sample preview

## Tests

Run the current unit tests:

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

The tests load the real encoder-only Sentence Transformer model. On the first run, the model is downloaded to the local Hugging Face cache. Later runs load it from cache.

The tests cover:

- Skill alias normalization
- Technical terms such as `C++`, `C#`, `.NET`, and `Node.js`
- Partial-word false positive prevention
- Matched and missing skill comparison
- Cosine similarity calculation
- Semantic score conversion
- Combined resume-job matching score
- Evaluation metrics such as Pearson and Spearman correlation

## Current Matching Pipeline

The current code can run the first complete NLP matching pipeline:

```text
Resume text + job description text
        |
        v
Encoder-only embedding similarity
        |
        v
Cosine similarity score
        |
        v
Dictionary-based skill extraction
        |
        v
Skill match score
        |
        v
Overall suitability score
        |
        v
HR-oriented explanation
```

Core modules:

- `src/similarity.py`: encoder-only model loading, embedding generation, cosine similarity
- `src/skill_extractor.py`: skill extraction and alias normalization
- `src/matcher.py`: combined semantic and skill-based scoring
- `src/skill_weights.py`: weighted skill scoring for technical, analytical, domain, basic tool, and general skills
- `src/recommender.py`: match category, HR evaluation, interview focus, and candidate suggestions
- `src/evaluation.py`: label mapping, correlation metrics, and threshold evaluation

Current score formula:

```text
overall_score =
  0.80 * semantic_similarity_score +
  0.20 * weighted_skill_match_score
```

Technical and role-specific skills receive more weight than broad general skills. General-only skill matches are capped so that soft-skill overlap alone does not produce a high suitability score.

Run real encoder-based matching on dataset samples:

```bash
.\.venv\Scripts\python.exe scripts/run_sample_matching.py --split test --limit 3
```

Run a balanced evaluation sample:

```bash
.\.venv\Scripts\python.exe scripts/evaluate_matching.py --split test --limit 60 --sample-mode balanced
```

The evaluation reports:

- Average semantic score by label
- Average skill match score by label
- Average overall score by label
- Pearson and Spearman correlation with dataset labels
- Threshold-based label prediction accuracy

Run qualitative error analysis:

```bash
.\.venv\Scripts\python.exe scripts/analyze_errors.py --split test --limit 60 --sample-mode balanced --top-n 2
```

The error analysis reports:

- `Good Fit` examples with unexpectedly low scores
- `No Fit` examples with unexpectedly high scores
- Extreme `Potential Fit` examples
- Matched and missing skills for each inspected case

The matcher output also includes:

- `match_category`
- `hr_evaluation`
- `interview_focus`
- `candidate_suggestions`

The match category is cautious when semantic similarity is high but key skills are missing. For example, the system can return `Strong Semantic Match with Skill Gaps` instead of a plain `Strong Match`.

Run top-k ranking evaluation from saved evaluation results:

```bash
.\.venv\Scripts\python.exe scripts/evaluate_ranking.py --csv data\processed\final_test_evaluation.csv
```

Final evaluation results are summarized in `RESULTS.md`.

## Sentence Transformer Fine-Tuning

The baseline system uses a pretrained encoder directly. Fine-tuning adapts the encoder to the resume-job matching dataset.

Training objective:

```text
No Fit        -> 0.0 cosine similarity target
Potential Fit -> 0.5 cosine similarity target
Good Fit      -> 1.0 cosine similarity target
```

This keeps the project focused on encoder-only NLP: the model learns to place matching resume-job pairs closer in embedding space and poor pairs farther apart.

Run a GPU-friendly balanced training pass:

```bash
.\.venv\Scripts\python.exe scripts/train_sentence_transformer.py --device cuda --train-limit 900 --eval-limit 300 --epochs 1 --batch-size 8
```

Run full-split fine-tuning:

```bash
.\.venv\Scripts\python.exe scripts/train_sentence_transformer.py --model sentence-transformers/all-MiniLM-L6-v2 --device cuda --train-limit 0 --eval-limit 0 --epochs 2 --batch-size 8 --output-dir models/resume-job-minilm-finetuned-2epoch
```

The fine-tuned model is saved under:

```text
models/resume-job-minilm-finetuned-2epoch/
```

Use it in existing evaluation scripts by passing the saved model path, or omit `--model` after the local fine-tuned model exists:

```bash
.\.venv\Scripts\python.exe scripts/evaluate_matching.py --model models/resume-job-minilm-finetuned-2epoch --split test --limit 300 --sample-mode balanced
```

## React Vite App

The primary UI is now a React Vite dashboard backed by FastAPI. The Python NLP pipeline remains unchanged behind the API.

Start the FastAPI backend:

```bash
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --port 8000
```

Start the React frontend in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

Backend endpoints:

```text
GET  /health
GET  /sample
POST /analyze
```

The backend loads the fine-tuned model once and reuses it for analysis requests.

## Streamlit App

Streamlit is kept as an optional Python-only demo interface.

Run the local interface:

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app provides:

- Resume text input
- Job description text input
- Overall, semantic, weighted skill, and raw skill scores
- Matched, missing, resume, and job skill lists
- HR evaluation
- Interview focus
- Candidate suggestions

## Development Roadmap

1. Load and inspect the dataset.
2. Build text preprocessing utilities.
3. Create a skill dictionary in `data/skills.json`.
4. Implement dictionary-based skill extraction.
5. Add encoder-only Sentence Transformer similarity.
6. Combine semantic similarity and skill match ratio into a final score.
7. Evaluate matching quality with dataset labels.
8. Generate HR-oriented explanations.
9. Build a Streamlit user interface.

## Minimum Successful Version

The project is considered successful when it can:

- Accept resume text and job description text
- Calculate semantic similarity
- Extract matched and missing skills
- Produce an overall suitability score
- Generate a clear HR-oriented explanation
- Report evaluation metrics
- Run through a simple user interface
