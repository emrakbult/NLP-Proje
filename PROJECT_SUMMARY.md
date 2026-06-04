# Project Summary: HR-Oriented Explainable Resume-Job Matching System

## Project Overview

This project is an explainable Natural Language Processing system for human resources use cases.

The system compares a candidate resume with a job description and produces an interpretable match result. It does not only return a similarity score. It explains the result by showing matched skills, missing skills, negated or unclear skill evidence, semantic similarity, neural fit prediction, overall suitability, and a deterministic match category.

The HR scenario is the application domain. The technical focus is encoder-only transformer modeling, cosine similarity, supervised training, validation-based checkpoint selection, skill extraction, and explainable text matching.

## Final System Capabilities

The final system can:

- Accept resume text and job description text
- Accept PDF, DOCX, TXT, and Markdown uploads for both resume and job description
- Encode both texts with a custom MiniLM-based bi-encoder
- Produce a cosine-based semantic score
- Produce a 3-class neural fit prediction: `No Fit`, `Potential Fit`, or `Good Fit`
- Extract skills from resume and job description text
- Validate resume skill mentions with a model-based evidence classifier
- Separate positive, negated, and unclear resume skill evidence
- Normalize skill aliases into canonical skill names
- Identify matched skills
- Identify missing or unclear required skills
- Calculate a weighted skill match score
- Calculate an overall suitability score
- Run through a React Vite dashboard backed by FastAPI
- Generate experiment reports and plots

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

Dataset size:

| Split | Rows |
|---|---:|
| Train | 6,241 |
| Test | 1,759 |
| Total | 8,000 |

The train split was divided into stratified train and validation subsets. The original test split remained untouched for final evaluation.

## NLP Architecture

The final runtime model is:

```text
models/resume-job-biencoder-optimal/
```

It is initialized from local pretrained MiniLM weights:

```text
models/base-minilm/
```

Architecture:

- Shared MiniLM encoder for resume and job text
- Mean pooling
- Projection head: `384 -> 256 -> 128`
- L2 normalization
- Cosine similarity with learnable temperature
- Pair classification head using resume embedding, job embedding, absolute difference, and elementwise product

The system also includes a second encoder-only classifier for sentence-level skill evidence validation:

```text
models/skill-evidence-minilm-classifier/
```

It classifies resume skill mentions as:

- `positive`
- `negated`
- `unclear`

This prevents statements such as `I do not have AWS experience` from being counted as positive AWS evidence.

## Training And Model Selection

The custom bi-encoder was trained for 25 epochs on GPU.

Loss function:

```text
total_loss = 0.6 * cosine_mse_loss + 0.4 * classification_cross_entropy
```

Every epoch logged:

- train total loss
- train cosine loss
- train classification loss
- validation total loss
- validation cosine loss
- validation classification loss
- validation accuracy
- validation Pearson correlation
- validation Spearman correlation

The optimal checkpoint was selected by lowest validation total loss.

Selected checkpoint:

| Selected Epoch | Validation Total Loss | Validation Accuracy |
|---:|---:|---:|
| 22 | 0.2558 | 0.8342 |

Training artifacts:

```text
reports/biencoder_training_config.json
reports/biencoder_training_history.csv
reports/biencoder_optimal_selection.json
```

## Final Evaluation

Final test split:

```text
1,759 records
```

Final custom bi-encoder metrics:

| Metric | Value |
|---|---:|
| Classification accuracy | 0.5225 |
| Semantic Pearson | 0.2024 |
| Semantic Spearman | 0.2150 |

Historical comparison:

| Model | Semantic Pearson | Semantic Spearman | Accuracy Metric |
|---|---:|---:|---:|
| Baseline local MiniLM | 0.1121 | 0.1078 | 0.4139 |
| New optimal custom bi-encoder | 0.2024 | 0.2150 | 0.5225 |

Interpretation:

- The new custom bi-encoder is stronger on direct 3-class fit prediction.
- The UI uses only the selected final model, while model comparison is documented in reports and plots.

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

The final category is intentionally cautious. If a resume is semantically close to a job description but misses role-specific requirements such as AWS or Docker, the system can report:

```text
Strong Semantic Match with Skill Gaps
```

This makes the prototype more suitable as an HR decision-support and screening tool, not as an automated hiring decision system.

## User Interface

The primary user interface is a React Vite dashboard:

```text
frontend/
```

The React frontend communicates with a FastAPI backend:

```text
api.py
```

The app provides:

- PDF/DOCX/TXT/Markdown upload for resume and job description
- Resume text input
- Job description text input
- Example loading
- Overall score
- Semantic score
- Neural fit prediction
- Weighted skill score
- Raw skill score
- Match category
- Matched and missing skills
- Negated and unclear skill mentions
- Skill evidence sentences

Model comparison is no longer shown in the UI. It belongs to `RESULTS.md` and `reports/`.

## Main Project Files

| File | Purpose |
|---|---|
| `IDEA.md` | Concept and project idea |
| `PLAN.md` | Project plan |
| `README.md` | Setup, usage, and project structure |
| `RESULTS.md` | Final evaluation results and analysis |
| `PROJECT_SUMMARY.md` | Completed project summary |
| `api.py` | FastAPI backend |
| `frontend/src/App.tsx` | React user interface |
| `src/neural_matcher.py` | Custom bi-encoder architecture and runtime |
| `src/neural_training.py` | Dataset, loss, epoch metrics, and checkpoint selection |
| `src/similarity.py` | Model loading and cosine utilities |
| `src/matcher.py` | Final matching pipeline |
| `src/skill_extractor.py` | Skill extraction and normalization |
| `src/skill_evidence.py` | Sentence-level skill evidence classification |
| `src/skill_weights.py` | Weighted skill scoring |
| `src/recommender.py` | Deterministic match category rules |
| `scripts/train_biencoder.py` | Custom bi-encoder training pipeline |
| `scripts/evaluate_biencoder.py` | Test evaluation and model comparison reports |
| `scripts/plot_training_history.py` | Training and evaluation plot generation |

## Final Status

The project is ready for demonstration as an NLP course project.

Completed work:

- Real dataset downloaded and used
- Local pretrained MiniLM model stored
- Custom bi-encoder architecture implemented
- 25-epoch GPU training completed
- Optimal checkpoint selected by validation loss
- Test evaluation completed
- Training and evaluation plots generated
- React Vite UI implemented
- FastAPI backend implemented
- MarkItDown upload implemented
- Skill evidence classifier trained and integrated
- Skill weighting and cautious match category rules implemented
- Tests updated for the new architecture

Final runtime model:

```text
models/resume-job-biencoder-optimal/
```
