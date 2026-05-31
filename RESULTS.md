# Results: HR-Oriented Explainable Resume-Job Matching System

## Final Model

The final semantic encoder is the fine-tuned Sentence Transformer model:

```text
models/resume-job-minilm-finetuned-2epoch/
```

Base model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Fine-tuning objective:

```text
No Fit        -> 0.0 cosine similarity target
Potential Fit -> 0.5 cosine similarity target
Good Fit      -> 1.0 cosine similarity target
```

The model was fine-tuned with `CosineSimilarityLoss`, keeping the project focused on encoder-only semantic representation and cosine similarity.

## Dataset

Main dataset:

```text
cnamuangtoun/resume-job-description-fit
```

Local files:

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

Test label distribution:

| Label | Count |
|---|---:|
| No Fit | 857 |
| Potential Fit | 444 |
| Good Fit | 458 |

## Fine-Tuning Result

Training configuration:

| Setting | Value |
|---|---|
| Device | CUDA GPU |
| GPU | NVIDIA GeForce GTX 1060 |
| Epochs | 2 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Max sequence length | 256 |

Full fine-tuning evaluator result:

| Metric | Before Fine-Tuning | After Fine-Tuning |
|---|---:|---:|
| Pearson correlation | 0.1110 | 0.4075 |
| Spearman correlation | 0.1065 | 0.3994 |

This shows that supervised fine-tuning significantly improved alignment between cosine similarity and the dataset fit labels.

The 2-epoch model was selected over the 1-epoch model because it improved the overall evaluation metrics and threshold-based accuracy while keeping the same architecture and training objective.

| Model | Pearson Semantic | Spearman Semantic | Pearson Overall | Spearman Overall | Accuracy |
|---|---:|---:|---:|---:|---:|
| 1 epoch | 0.4029 | 0.3780 | 0.3734 | 0.3480 | 0.4741 |
| 2 epochs | 0.4022 | 0.3993 | 0.3815 | 0.3766 | 0.4821 |

## Full Test Evaluation

Command:

```bash
.\.venv\Scripts\python.exe scripts\evaluate_matching.py --split test --limit 0 --sample-mode balanced --batch-size 32 --save-csv data\processed\final_test_evaluation.csv
```

Evaluation output:

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

Average weighted skill match score by label:

| Label | Average Score |
|---|---:|
| No Fit | 20.82 |
| Potential Fit | 20.49 |
| Good Fit | 24.19 |

Average overall score by label:

| Label | Average Score |
|---|---:|
| No Fit | 25.57 |
| Potential Fit | 40.73 |
| Good Fit | 43.37 |

Interpretation:

- The fine-tuned semantic encoder separates `No Fit` from the other labels clearly.
- `Potential Fit` and `Good Fit` are closer to each other, which is expected because both can share many relevant resume-job signals.
- The skill score is useful for explanation, but it is less predictive than the fine-tuned semantic score on this dataset.
- The overall score combines semantic similarity and skill evidence, but semantic similarity remains the strongest quantitative signal.

## Ranking Evaluation

Command:

```bash
.\.venv\Scripts\python.exe scripts\evaluate_ranking.py --csv data\processed\final_test_evaluation.csv
```

Ranking by semantic score:

| K | Precision@K Good Fit | Precision@K Potential-or-Good Fit |
|---:|---:|---:|
| 50 | 0.6400 | 0.9600 |
| 100 | 0.4900 | 0.8600 |
| 200 | 0.3850 | 0.7850 |

Ranking by overall score:

| K | Precision@K Good Fit | Precision@K Potential-or-Good Fit |
|---:|---:|---:|
| 50 | 0.5200 | 0.8400 |
| 100 | 0.4700 | 0.8000 |
| 200 | 0.4100 | 0.7450 |

Interpretation:

- The semantic encoder is effective for ranking strong candidate-job pairs near the top.
- In the top 50 semantic results, 96% are either `Potential Fit` or `Good Fit`.
- The overall score is more explainable because it includes skills, but semantic ranking performs better for pure retrieval.

## Qualitative Error Analysis

Command:

```bash
.\.venv\Scripts\python.exe scripts\analyze_errors.py --split test --limit 0 --sample-mode balanced --batch-size 32 --top-n 2
```

Main findings:

1. Some `Good Fit` examples receive low scores when the resume and job posting use different wording and the explicit skill dictionary finds few or no shared skills.
2. Some `No Fit` examples receive high scores when the candidate has many overlapping technical keywords with the job posting, such as Java, JavaScript, HTML, and CSS.
3. Some `Potential Fit` examples receive high semantic scores because the resume and job are in the same professional domain, even when the fit label is not `Good Fit`.
4. Broad tools such as Excel can still create strong skill overlap in non-technical roles, so skill weighting should remain conservative.

These errors are useful for the project because they show why explainability is necessary. The score alone is not enough; HR should also inspect matched skills, missing skills, and the generated evaluation notes.

## Skill Extraction Evaluation

The dataset provides resume-job fit labels, but it does not provide gold-standard skill annotations. Because of this, dataset-level Precision, Recall, and F1 for skill extraction cannot be calculated directly from the main dataset.

Skill extraction is evaluated through:

- Unit tests for alias matching and partial-word false positive prevention
- Manual qualitative inspection in error analysis
- Matched and missing skill explanations in the Streamlit app

The current skill extraction module uses dictionary matching for candidate skill spans and a model-based evidence classifier for resume-side validation. This prevents negated or weak mentions from being counted as matched skills.

## Skill Evidence Classifier Result

The skill evidence classifier was trained on a manually curated dataset:

```text
data/skill_evidence/skill_evidence_dataset.csv
```

Dataset size:

| Label | Rows |
|---|---:|
| Positive | 67 |
| Negated | 67 |
| Unclear | 67 |
| Total | 201 |

Split:

| Split | Rows |
|---|---:|
| Train | 165 |
| Validation | 18 |
| Test | 18 |

Classifier configuration:

| Setting | Value |
|---|---|
| Base model | microsoft/MiniLM-L12-H384-uncased |
| Output path | models/skill-evidence-minilm-classifier/ |
| Epochs | 12 |
| Batch size | 16 |
| Device | CUDA GPU |

Final result:

| Metric | Value |
|---|---:|
| Validation accuracy | 1.0000 |
| Test accuracy | 0.9444 |

This dataset is intentionally small and controlled, so these metrics should not be interpreted as production-level generalization. The value for the course project is that the pipeline now uses a second encoder-only model to classify sentence-level evidence as `positive`, `negated`, or `unclear`.

## Final Status

The project now includes:

- Real dataset loading
- Encoder-only semantic matching
- GPU fine-tuned Sentence Transformer
- Cosine similarity scoring
- Skill extraction and normalization
- Model-based skill evidence classification
- Matched and missing skill explanations
- Negated and unclear skill explanations
- Weighted skill scoring
- MarkItDown file upload for PDF/DOCX/TXT/Markdown inputs
- HR-oriented recommendations
- Full test evaluation
- Ranking evaluation
- Qualitative error analysis
- React Vite user interface backed by FastAPI
- Optional Streamlit user interface

The project is ready for demonstration and presentation.
