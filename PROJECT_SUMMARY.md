# Project Summary: HR-Oriented Explainable Resume-Job Matching System

## Project Overview

This project is an explainable Natural Language Processing system for human resources use cases.

The system compares a candidate resume with a job description and produces an interpretable match result. It does not only return a similarity score. It also explains the result by showing matched skills, missing skills, semantic similarity, overall suitability, and HR-oriented evaluation notes.

The project is designed as an NLP project first. The HR scenario is the application domain. The technical focus is encoder-only transformer embeddings, cosine similarity, supervised fine-tuning, skill extraction, and explainable text matching.

## Final System Capabilities

The final system can:

- Accept resume text and job description text
- Accept PDF, DOCX, TXT, and Markdown uploads for both resume and job description
- Encode both texts with a fine-tuned Sentence Transformer
- Calculate cosine similarity between resume and job embeddings
- Extract skills from resume and job description text
- Validate resume skill mentions with a model-based evidence classifier
- Separate positive, negated, and unclear resume skill evidence
- Normalize skill aliases into canonical skill names
- Identify matched skills
- Identify missing or unclear required skills
- Calculate a weighted skill match score
- Calculate an overall suitability score
- Produce cautious match categories when strong semantic similarity still has missing key skills
- Generate HR-oriented explanations
- Suggest technical interview focus areas
- Run through a React Vite dashboard backed by FastAPI
- Keep Streamlit as an optional Python-only demo interface
- Evaluate the model on a real resume-job fit dataset

## Dataset

Main dataset:

```text
cnamuangtoun/resume-job-description-fit
```

Local dataset files:

```text
data/raw/cnamuangtoun_resume_job_description_fit/train.csv
data/raw/cnamuangtoun_resume_job_description_fit/test.csv
```

Dataset columns:

- `resume_text`
- `job_description_text`
- `label`

Label classes:

- `No Fit`
- `Potential Fit`
- `Good Fit`

Dataset size:

| Split | Rows |
|---|---:|
| Train | 6,241 |
| Test | 1,759 |
| Total | 8,000 |

## NLP Approach

The project uses an encoder-only Sentence Transformer approach.

Basic semantic matching flow:

```text
Resume Text -------> Encoder-only Model -------> Resume Embedding
                                                    |
                                                    | Cosine Similarity
                                                    |
Job Text ----------> Encoder-only Model -------> Job Embedding
```

The first baseline used:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The final model is a supervised fine-tuned version of this encoder.

Final model path:

```text
models/resume-job-minilm-finetuned-2epoch/
```

The system also includes a second encoder-only classifier for sentence-level skill evidence validation.

Skill evidence classifier path:

```text
models/skill-evidence-minilm-classifier/
```

It classifies resume skill mentions as:

- `positive`
- `negated`
- `unclear`

This prevents statements such as `I do not have AWS experience` from being counted as positive AWS evidence.

## Fine-Tuning

The encoder was fine-tuned on the resume-job fit dataset.

Label-to-score mapping:

```text
No Fit        -> 0.0
Potential Fit -> 0.5
Good Fit      -> 1.0
```

Training objective:

```text
CosineSimilarityLoss
```

Training configuration:

| Setting | Value |
|---|---|
| Base model | sentence-transformers/all-MiniLM-L6-v2 |
| Final epochs | 2 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Max sequence length | 256 |
| Device | CUDA GPU |
| GPU | NVIDIA GeForce GTX 1060 |

The 2-epoch model was selected as the final model because it performed better than the 1-epoch model on overall evaluation metrics.

Comparison:

| Model | Pearson Overall | Spearman Overall | Accuracy |
|---|---:|---:|---:|
| 1 epoch | 0.3734 | 0.3480 | 0.4741 |
| 2 epochs | 0.3815 | 0.3766 | 0.4821 |

## Final Evaluation

Full test split evaluation:

```text
Test records: 1,759
```

Final evaluation metrics:

| Metric | Value |
|---|---:|
| Pearson semantic | 0.4022 |
| Spearman semantic | 0.3993 |
| Pearson overall | 0.3815 |
| Spearman overall | 0.3766 |
| Threshold-based accuracy | 0.4821 |

Average semantic score by label:

| Label | Average Score |
|---|---:|
| No Fit | 26.76 |
| Potential Fit | 45.79 |
| Good Fit | 48.17 |

Average overall score by label:

| Label | Average Score |
|---|---:|
| No Fit | 25.57 |
| Potential Fit | 40.73 |
| Good Fit | 43.37 |

## Ranking Results

Ranking by semantic score:

| K | Precision@K Good Fit | Precision@K Potential-or-Good Fit |
|---:|---:|---:|
| 50 | 0.6200 | 0.9600 |
| 100 | 0.4900 | 0.8600 |
| 200 | 0.3850 | 0.7850 |

Ranking by overall score:

| K | Precision@K Good Fit | Precision@K Potential-or-Good Fit |
|---:|---:|---:|
| 50 | 0.5200 | 0.8400 |
| 100 | 0.4700 | 0.8000 |
| 200 | 0.4100 | 0.7450 |

## Explainability

The system explains the score with skill-level evidence.

For each resume-job pair, it reports:

- Resume skills
- Job description skills
- Matched skills
- Missing skills
- Negated skill mentions
- Unclear skill mentions
- Evidence sentence for each detected resume skill mention
- Weighted skill match score
- HR evaluation
- Interview focus suggestions
- Candidate improvement suggestions

The final category is intentionally cautious. For example, if a resume is semantically close to a job description but misses role-specific requirements such as AWS or Docker, the system can report:

```text
Strong Semantic Match with Skill Gaps
```

This makes the prototype more suitable as an HR decision-support and screening tool, not as an automated hiring decision system.

The skill extraction module uses a dictionary stored in:

```text
data/skills.json
```

It supports aliases and technical terms such as:

- Python
- SQL
- Git
- Docker
- Kubernetes
- AWS
- C++
- C#
- .NET
- Node.js

Technical and role-specific skills receive higher weights than broad general skills. General skills such as communication or leadership are intentionally weighted lower so they do not overinflate the match score.

## User Interface

The primary user interface is now a React Vite dashboard:

```text
frontend/
```

The React frontend communicates with a FastAPI backend:

```text
api.py
```

Run the backend:

```bash
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --port 8000
```

Run the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

The project also keeps the original Streamlit app as an optional Python-only demo:

```text
app.py
```

Run command:

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens at:

```text
http://localhost:8501
```

The app provides:

- PDF/DOCX/TXT/Markdown upload for resume and job description
- Resume text input
- Job description text input
- Example loading
- Overall score
- Semantic score
- Weighted skill score
- Raw skill score
- Match category
- HR explanation
- Matched and missing skills
- Negated and unclear skill mentions
- Skill evidence sentences
- Interview focus
- Candidate suggestions

## Main Project Files

| File | Purpose |
|---|---|
| `IDEA.md` | Concept and project idea |
| `PLAN.md` | Detailed implementation plan |
| `README.md` | Setup, usage, and project structure |
| `RESULTS.md` | Final evaluation results and analysis |
| `PROJECT_SUMMARY.md` | Completed project summary |
| `app.py` | Streamlit user interface |
| `src/similarity.py` | Model loading, embeddings, cosine similarity |
| `src/matcher.py` | Final matching pipeline |
| `src/skill_extractor.py` | Skill extraction and normalization |
| `src/skill_evidence.py` | Sentence-level skill evidence classification |
| `src/skill_weights.py` | Weighted skill scoring |
| `src/recommender.py` | HR-oriented explanations |
| `src/evaluation.py` | Evaluation metrics |
| `src/fine_tuning.py` | Fine-tuning helpers |
| `scripts/train_sentence_transformer.py` | Encoder fine-tuning script |
| `scripts/train_skill_evidence_classifier.py` | Skill evidence classifier training script |
| `scripts/evaluate_matching.py` | Full evaluation script |
| `scripts/evaluate_ranking.py` | Top-k ranking evaluation |
| `scripts/analyze_errors.py` | Qualitative error analysis |

## Final Status

The project is complete and ready for demonstration.

Completed work:

- Real dataset downloaded and used
- Baseline encoder matching implemented
- Skill extraction implemented
- Weighted skill scoring implemented
- HR explanations implemented
- MarkItDown upload implemented for resume and job description files
- Skill evidence classifier trained and integrated
- Streamlit app implemented
- GPU-enabled PyTorch configured
- Sentence Transformer fine-tuned on the full dataset
- 1 epoch and 2 epoch models compared
- 2 epoch model selected as final
- Full test evaluation completed
- Ranking evaluation completed
- Qualitative error analysis completed
- Tests pass successfully

Final verification:

```text
Default model: models/resume-job-minilm-finetuned-2epoch
Skill evidence model: models/skill-evidence-minilm-classifier
Tests: 50 passed
```
