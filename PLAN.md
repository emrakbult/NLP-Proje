# Project Plan: HR-Oriented Explainable Resume-Job Matching System

## 1. Project Definition

This project is an explainable Natural Language Processing system designed for human resources use cases.

The system compares a candidate's resume or CV with a job posting and evaluates how suitable the candidate is for the role. The goal is not only to calculate a similarity score between two texts. The system must also explain why the candidate matches or does not match the job requirements.

The system should answer the following questions:

- How suitable is the candidate for the job posting?
- Which skills in the resume match the job requirements?
- Which required skills are missing or not clearly visible in the resume?
- What should HR pay attention to while evaluating the candidate?
- Which topics should be checked during the technical interview?

## 2. Main Objective

The main objective is to build an NLP-based decision support tool for human resources teams.

The system will take two inputs:

- Resume or CV text
- Job posting text

After analyzing these inputs, the system will produce:

- Overall suitability score
- Matched skills
- Missing or unclear skills
- Short explanation of the score
- HR-oriented candidate evaluation
- Technical interview focus suggestions

## 3. HR-Oriented Perspective

The project will be designed from a human resources perspective.

This means the output should be understandable for HR staff, not only for technical users. The system should not only display raw model scores. Instead, it should provide clear and explainable results that help HR understand the candidate's strengths and gaps.

Example output:

```text
Overall Suitability: 72%

Matched Skills:
Python, SQL, Git

Missing or unclear skills:
Docker, Kubernetes, AWS

HR Evaluation:
The candidate appears suitable for basic software development tasks.
However, cloud and container technologies required by the job posting are not clearly visible in the resume.
Docker, Kubernetes, and AWS experience should be checked during the technical interview.
```

## 4. NLP Approach

The project will use an encoder-only model approach.

The main model will be a Sentence Transformer. This model converts the resume and job posting texts into numerical vectors called embeddings. These embeddings represent the semantic meaning of the texts. After both texts are encoded, cosine similarity will be used to measure how close they are.

Basic flow:

```text
Resume Text -------> Encoder-only Model -------> Resume Embedding
                                                    |
                                                    | Cosine Similarity
                                                    |
Job Posting Text --> Encoder-only Model -------> Job Embedding
```

This approach is suitable because the system does not need text generation as its main task. The main goal is to represent texts semantically and compare them.

The project should be presented primarily as a Natural Language Processing project, not as a simple HR web application. The HR scenario is the application domain, while the technical focus is semantic text representation, encoder-only transformer embeddings, cosine similarity, skill extraction, and explainable text matching.

Main NLP-focused parts:

- Text preprocessing for resume and job description documents
- Encoder-only Sentence Transformer representation
- Resume and job description embedding generation
- Cosine similarity-based semantic matching
- Skill and keyphrase extraction from unstructured text
- Skill normalization and alias matching
- Explainable comparison between extracted resume skills and job requirements
- Evaluation of both extraction quality and matching quality

Approximate project focus:

```text
50% Semantic similarity with encoder-only embeddings
30% Skill extraction, normalization, and explainability
10% Evaluation and error analysis
10% User interface and demonstration
```

## 5. Model Selection

The first model option will be:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Advantages:

- Lightweight
- Fast
- Easy to install and use
- Suitable for semantic similarity tasks
- Good enough for the first working prototype

If Turkish-English mixed texts or multilingual inputs become important, the following model can be used:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

The first version will not use a decoder or encoder-decoder model. The main model type will be an encoder-only Sentence Transformer.

### 5.1 Supervised Fine-Tuning Track

After the pretrained baseline is working, the encoder can be fine-tuned on the resume-job fit dataset.

Fine-tuning will still use an encoder-only Sentence Transformer. The goal is not to generate text. The goal is to improve the embedding space so that suitable resume-job pairs become closer and unsuitable pairs become farther apart.

The dataset labels will be converted into cosine similarity targets:

```text
No Fit        -> 0.0
Potential Fit -> 0.5
Good Fit      -> 1.0
```

The fine-tuning objective will use `CosineSimilarityLoss`. This directly matches the project goal because the final matching system is based on semantic embeddings and cosine similarity.

Fine-tuning workflow:

1. Load `train.csv` and `test.csv`.
2. Convert each row into a resume-job text pair.
3. Convert dataset labels into numeric similarity targets.
4. Fine-tune `sentence-transformers/all-MiniLM-L6-v2`.
5. Evaluate baseline vs fine-tuned model with Pearson/Spearman correlation and score distribution by label.
6. Use the fine-tuned model path in the existing matcher and evaluation scripts.

The fine-tuned model will be saved locally under:

```text
models/resume-job-minilm-finetuned-2epoch/
```

GPU training is preferred. On the current Windows environment, the venv should use CUDA PyTorch instead of CPU-only PyTorch:

```text
torch 2.12.0+cu126
```

## 6. Main System Components

The system will consist of the following components:

1. Text input
2. Text preprocessing
3. Skill extraction
4. Semantic similarity calculation
5. Score calculation
6. Explainable result generation
7. User interface

## 7. Input Structure

The first version will use direct text input.

The user will provide:

- Resume text
- Job posting text

PDF upload is not required for the first version. Starting with text input keeps the first prototype simple and allows the core NLP logic to be tested more easily.

Future input options:

- PDF resume upload
- PDF job posting upload
- DOCX file support

## 8. Text Preprocessing

The preprocessing step prepares raw text for analysis.

Basic preprocessing operations:

- Remove unnecessary extra spaces
- Normalize line breaks
- Apply lowercase normalization for matching
- Handle punctuation carefully
- Preserve technical expressions

Important note:

Technical skills may contain special characters. For example, terms such as `C++`, `C#`, `.NET`, and `Node.js` should not be damaged during preprocessing.

Therefore, preprocessing should be simple and controlled rather than overly aggressive.

## 9. Skill Dictionary

The project will use a skill dictionary stored in `data/skills.json`.

This dictionary will contain canonical skill names and their alternative spellings or aliases.

Example:

```json
{
  "Python": ["python"],
  "SQL": ["sql"],
  "Git": ["git", "github", "gitlab"],
  "Docker": ["docker"],
  "Kubernetes": ["kubernetes", "k8s"],
  "AWS": ["aws", "amazon web services"],
  "PostgreSQL": ["postgresql", "postgres", "postgre sql"],
  "REST API": ["rest api", "restful api", "restful services"]
}
```

The skill dictionary is important for explainability. The encoder model can show that two texts are semantically close, but dictionary-based skill extraction makes it possible to show exactly which skills matched and which skills are missing.

## 10. Skill Extraction

The first version will use rule-based skill extraction.

The system will follow these steps:

1. Load the skill dictionary.
2. Search for skill aliases in the resume text.
3. Search for skill aliases in the job posting text.
4. Normalize found aliases to canonical skill names.
5. Compare resume skills with job posting skills.

Main outputs:

- `resume_skills`
- `job_skills`
- `matched_skills`
- `missing_skills`

Example:

```text
Resume Skills:
Python, SQL, Git

Job Skills:
Python, SQL, Git, Docker, AWS

Matched Skills:
Python, SQL, Git

Missing Skills:
Docker, AWS
```

## 11. Semantic Similarity Calculation

The semantic similarity module measures how close the resume and job posting are in meaning.

Steps:

1. Load the Sentence Transformer model.
2. Convert the resume text into an embedding.
3. Convert the job posting text into an embedding.
4. Calculate cosine similarity between the two embeddings.
5. Convert the similarity value into a percentage score.

Example:

```text
Cosine similarity: 0.76
Semantic similarity score: 76%
```

This score represents the general semantic closeness between the resume and the job posting.

## 12. Skill Match Ratio

The skill match ratio measures how many of the job posting skills are found in the resume.

Formula:

```text
skill_match_ratio = number_of_matched_skills / total_number_of_job_skills
```

Example:

```text
Job skills: Python, SQL, Git, Docker, AWS
Matched skills: Python, SQL, Git

skill_match_ratio = 3 / 5 = 0.60
```

Skill match score:

```text
60%
```

## 13. Overall Score Calculation

The overall suitability score will combine two main signals:

- Semantic similarity score
- Skill match score

Initial formula:

```text
overall_score =
  0.80 * semantic_similarity_score +
  0.20 * weighted_skill_match_score
```

Example:

```text
semantic_similarity_score = 76
weighted_skill_match_score = 60

overall_score = 0.80 * 76 + 0.20 * 60
overall_score = 72.8
```

Final result:

```text
Overall Suitability: 73%
```

The current formula is semantic-heavy because early evaluation showed that raw skill overlap can overvalue broad skills. The skill match score is weighted so technical and role-specific skills contribute more than general soft skills.

## 14. Explainable Result Generation

Explainability is one of the most important parts of the project.

The explanation should include:

- Why the candidate is suitable
- Which skills support the match
- Which skills are missing
- What HR should pay attention to
- Which topics should be checked during the technical interview

Example explanation:

```text
The candidate matches the job posting in Python, SQL, and Git.
However, Docker and AWS are required in the job posting but are not clearly found in the resume.
The candidate may still be suitable for junior or mid-level software development tasks,
but cloud and container experience should be checked during the technical interview.
```

## 15. HR Evaluation Logic

The first version will use rule-based HR evaluation.

Example rules:

- If the overall score is 80% or higher, the candidate is a strong match.
- If the overall score is between 60% and 79%, the candidate is a partial match.
- If the overall score is between 40% and 59%, the candidate is a weak match.
- If the overall score is below 40%, the candidate is not recommended for this role.
- If critical missing skills exist, they should be clearly mentioned in the explanation.

Example categories:

```text
80-100%: Strong Match
60-79%: Partial Match
40-59%: Weak Match
0-39%: Not Recommended
```

## 16. User Interface

The first interface is built with Streamlit.

The interface will include:

- Resume text input area
- Job posting text input area
- Analyze button
- Overall suitability score
- Semantic similarity score
- Skill match score
- Matched skills section
- Missing skills section
- HR evaluation section
- Technical interview focus section

Streamlit is preferred because it enables fast development, simple execution, and clear demonstration for NLP projects.

Run command:

```text
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## 17. Suggested File Structure

```text
NLP-Proje/
|-- IDEA.md
|-- PLAN.md
|-- README.md
|-- requirements.txt
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
|   `-- evaluate_matching.py
|-- src/
|   |-- __init__.py
|   |-- data_loader.py
|   |-- evaluation.py
|   |-- preprocessing.py
|   |-- skill_extractor.py
|   |-- skill_weights.py
|   |-- similarity.py
|   |-- matcher.py
|   `-- recommender.py
`-- tests/
    |-- conftest.py
    |-- test_evaluation.py
    |-- test_recommender.py
    |-- test_skill_extractor.py
    |-- test_similarity.py
    `-- test_matcher.py
```

## 18. Module Responsibilities

### `src/data_loader.py`

Responsible for loading and validating local dataset files.

Planned functions:

- `validate_dataset_file(path)`
- `load_records(path, limit=None)`
- `summarize_dataset(dataset_dir)`

### `src/preprocessing.py`

Responsible for text cleaning and normalization.

Planned functions:

- `clean_text(text)`
- `normalize_whitespace(text)`

### `src/skill_extractor.py`

Responsible for extracting skills from resume and job posting texts.

Planned functions:

- `load_skills(path)`
- `extract_skills(text, skill_dictionary)`
- `compare_skills(resume_skills, job_skills)`

### `src/skill_weights.py`

Responsible for assigning different contribution weights to different skill types.

Planned functions:

- `get_skill_weight(skill)`
- `total_skill_weight(skills)`
- `has_role_specific_skill(skills)`

### `src/similarity.py`

Responsible for loading the encoder-only model and calculating cosine similarity.

Planned functions:

- `load_model(model_name)`
- `encode_text(text, model)`
- `encode_texts(texts, model, batch_size)`
- `calculate_cosine_similarity(resume_text, job_text, model)`
- `calculate_semantic_similarity_score(resume_text, job_text, model)`

### `src/matcher.py`

Responsible for combining semantic similarity and skill matching.

Planned functions:

- `calculate_skill_match_score(matched_skills, job_skills)`
- `calculate_final_score(semantic_score, skill_score)`
- `match_resume_to_job(resume_text, job_text)`

### `src/evaluation.py`

Responsible for label mapping and evaluation metrics.

Planned functions:

- `label_to_score(label)`
- `pearson_correlation(values_a, values_b)`
- `spearman_correlation(values_a, values_b)`
- `average_scores_by_label(labels, scores)`
- `classification_accuracy(true_labels, predicted_labels)`

### `src/recommender.py`

Responsible for generating HR-oriented explanations and recommendations.

Planned functions:

- `classify_match(overall_score)`
- `generate_recommendation(overall_score, semantic_score, skill_match_score, matched_skills, missing_skills)`
- `generate_hr_evaluation(final_score, matched_skills, missing_skills)`
- `generate_interview_focus(missing_skills)`

### `app.py`

Responsible for the Streamlit user interface.

## 19. Development Steps

### Step 1: Create Project Structure

Create the main folders and starter files:

- `src/`
- `data/`
- `data/raw/`
- `data/processed/`
- `data/examples/`
- `scripts/`
- `tests/`
- `requirements.txt`
- `README.md`
- `app.py`

### Step 2: Define Requirements

Initial Python libraries:

```text
streamlit
sentence-transformers
scikit-learn
pandas
numpy
pypdf
pytest
```

### Step 3: Create Skill Dictionary

Create `data/skills.json`.

The first skill list should include:

- Programming languages
- Databases
- Web technologies
- Cloud platforms
- DevOps tools
- Data analysis tools
- Machine learning skills
- General professional skills

### Step 4: Implement Text Preprocessing

Write basic text cleaning functions.

This module will be used for both resume and job posting texts.

### Step 5: Implement Skill Extraction

Apply dictionary-based skill extraction.

Simple test example:

```text
Resume: "I know Python, SQL, and Git."
Job posting: "We are looking for a candidate who knows Python, SQL, Git, and Docker."

Matched: Python, SQL, Git
Missing: Docker
```

### Step 6: Implement Semantic Similarity

Load the Sentence Transformer model and calculate cosine similarity between the resume and the job posting.

### Step 7: Implement Overall Matching Logic

Combine the following outputs in one module:

- Semantic similarity score
- Skill match score
- Overall suitability score
- Matched skills
- Missing skills

### Step 8: Generate Explanation and HR Evaluation

Generate HR-oriented explanation based on the score and missing skills.

Example:

```text
The candidate is a partial match.
Core software development skills are suitable, but Docker and AWS experience are not clearly visible in the resume.
These topics should be checked during the technical interview.
```

### Step 9: Build Streamlit Interface

Create the user interface.

The interface will include two text input areas and a result section.

### Step 10: Test With Example Data

Prepare example resume and job posting texts.

At least three scenarios should be tested:

- Strong match
- Partial match
- Weak match

## 20. Evaluation Plan

The project will not be evaluated with a single metric because it has two different NLP responsibilities:

- Extracting useful information from resume and job description texts
- Producing a general resume-job matching score

Therefore, the evaluation will be performed at two levels:

- Information extraction performance
- Matching and ranking performance

### Skill Extraction Evaluation

Skill extraction evaluation will measure how accurately the system extracts skills from resumes and job descriptions.

Metrics:

- Precision
- Recall
- F1-score

Purpose:

To measure whether the extracted skills are correct and complete.

Definitions:

- Precision: among the skills extracted by the system, how many are actually correct?
- Recall: among the skills that should have been extracted, how many did the system find?
- F1-score: harmonic mean of precision and recall

Example:

```text
Gold skills:
Python, SQL, Git, Docker

Extracted skills:
Python, SQL, AWS

Correctly extracted:
Python, SQL

Precision = 2 / 3
Recall = 2 / 4
F1-score = harmonic mean of precision and recall
```

### Matching Performance Evaluation

Matching evaluation will measure how well the system compares a resume with a job description.

Metrics:

- Cosine similarity-based suitability score
- Pearson or Spearman correlation with human scores
- Top-k ranking performance
- Qualitative error analysis

Purpose:

To measure whether the generated suitability score agrees with human judgment and whether the system can rank better matches higher.

#### Cosine Similarity-Based Suitability Score

The encoder-only Sentence Transformer model will generate embeddings for the resume and job description. Cosine similarity will be used to calculate their semantic closeness.

```text
resume_embedding = encoder(resume_text)
job_embedding = encoder(job_description_text)

semantic_similarity = cosine_similarity(resume_embedding, job_embedding)
```

This score will show the semantic closeness between the resume and job description.

#### Correlation With Human Evaluation

The system score will be compared with human or label-based evaluation.

Possible mappings:

```text
Good Fit -> 2
Potential Fit -> 1
No Fit -> 0
```

The relationship between the system score and human/label score will be measured using:

- Pearson correlation
- Spearman correlation

Pearson correlation measures linear relationship. Spearman correlation measures whether the ranking order is similar.

#### Top-k Ranking Performance

Top-k ranking will be used when one resume is compared against multiple job descriptions or when one job description is compared against multiple resumes.

Example:

```text
Input:
One resume + 20 job descriptions

Expected:
The most suitable job descriptions should appear in the top-k results.
```

Possible metrics:

- Top-1 accuracy
- Top-3 accuracy
- Top-5 accuracy
- Mean Reciprocal Rank if needed

#### Qualitative Error Analysis

Some incorrect or weak examples will be manually analyzed.

The analysis will focus on:

- Skills that were incorrectly extracted
- Skills that were missed by the dictionary
- Cases where cosine similarity is high but skill match is weak
- Cases where skill match is high but semantic similarity is weak
- Confusing examples between `Good Fit` and `Potential Fit`
- Confusing examples between `Potential Fit` and `No Fit`

This will help explain system limitations and improve the skill dictionary or scoring formula.

### Final Evaluation Summary

The final report should include:

- Skill extraction precision, recall, and F1-score
- Average cosine similarity by label class
- Correlation between system score and dataset labels or manual scores
- Top-k ranking result if ranking experiments are included
- Qualitative error analysis examples

With these metrics, the project will be presented not only as a working prototype but as an evaluated and interpreted NLP application.

## 21. Test Scenarios

### Strong Match

The resume contains most of the skills required by the job posting.

Expected result:

- High overall score
- Many matched skills
- Few missing skills
- Positive HR evaluation

### Partial Match

The resume contains some important skills but misses several job requirements.

Expected result:

- Medium overall score
- Some matched skills
- Some missing skills
- Suggestion to check missing areas during the interview

### Weak Match

The resume contains very few skills related to the job posting.

Expected result:

- Low overall score
- Few matched skills
- Many missing skills
- Candidate is not strongly recommended for the role

## 22. Possible Future Improvements

After the first working version, the following features can be added:

- PDF resume upload
- DOCX resume upload
- Matching one resume against multiple job postings
- Ranking multiple resumes for one job posting
- Separating required and preferred skills
- Section-based resume analysis
- Education and experience matching
- Turkish-English mixed language support
- More advanced named entity recognition
- Cross-encoder reranking
- Exportable HR evaluation report

## 23. Minimum Successful Version

The project can be considered successful if it can:

- Accept resume text
- Accept job posting text
- Calculate semantic similarity using an encoder-only model
- Extract skills from both texts
- Show matched skills
- Show missing skills
- Calculate an overall suitability score
- Produce a short HR-oriented explanation
- Run through a simple user interface

## 24. Final Deliverables

The final project should include:

- Working Python application
- Streamlit interface
- Skill dictionary
- Example resume and job posting texts
- Explanation of the model and method
- Evaluation results
- Final report or presentation material

## 25. Suggested Timeline

### Week 1

- Create the project structure
- Define dependencies
- Prepare the first version of the skill dictionary
- Add example resume and job posting texts

### Week 2

- Implement text preprocessing
- Implement skill extraction
- Test matched and missing skill detection

### Week 3

- Add the encoder-only Sentence Transformer model
- Calculate cosine similarity
- Implement the overall score formula

### Week 4

- Implement the HR explanation module
- Build the Streamlit interface
- Test the system with example scenarios

### Week 5

- Fix errors and improve outputs
- Prepare evaluation metrics
- Organize results for report and presentation

## 26. Dataset Choice

The project will use a real-data-first strategy. Real data is preferred because it better represents actual resume language, job posting structure, incomplete skill descriptions, and noisy HR documents.

However, real resume data must be handled carefully because resumes may contain private information such as names, emails, phone numbers, addresses, and work history. Therefore, the project will only use public datasets and will avoid displaying or storing unnecessary personal information.

### Main Dataset

The recommended main dataset is:

```text
cnamuangtoun/resume-job-description-fit
```

This dataset is the preferred main dataset because it is larger and already contains labeled resume-job description pairs.

Useful fields:

- `resume_text`
- `job_description_text`
- `label`

This dataset will be used for:

- Developing the matching pipeline
- Training or evaluating resume-job fit classification
- Testing resume-job semantic similarity
- Evaluating whether the system separates strong, partial, and weak matches
- Creating demonstration examples

The label values represent different fit levels, such as:

- `Good Fit`
- `Potential Fit`
- `No Fit`

This makes the dataset useful for evaluating whether the final suitability score agrees with labeled fit categories.

### Real Resume Source

The resume texts in the selected datasets provide realistic resume-style language across different job categories.

If needed, the resume dataset can also be used separately to test skill extraction and resume parsing.

### Real Job Posting Source

For additional real job descriptions, the project can use:

```text
1.3M LinkedIn Jobs & Skills (2024)
```

This dataset contains real job postings scraped from LinkedIn and includes job-related skill information. It can be used to test the system on more realistic job descriptions, especially for software, data, cloud, and IT roles.

### Skill Dictionary Source

The first version will use a custom `data/skills.json` file. This dictionary will include common software, data, cloud, DevOps, and professional skills.

Later, it can be expanded with a public skill taxonomy such as:

```text
ESCO - European Skills, Competences, Qualifications and Occupations
```

ESCO can help improve skill coverage and standardization.

### Secondary Dataset

The secondary dataset is:

```text
batuhanmtl/job_resume_fit
```

This dataset is smaller, but it contains useful structured fields for skill-based evaluation.

Useful fields:

- `resume_text`
- `job_text`
- `category`
- `job_required_skills`
- `resume_skill_list`
- `ai_matched_skills`
- `ai_match_score`
- `skill_string_match_score`
- `fuzzy_match_score`

It will be used for:

- Testing skill extraction
- Comparing skill match scores
- Creating additional demo examples
- Checking whether our explanations align with listed matched skills

### Synthetic Dataset as Backup

Synthetic data will not be the main dataset. It will only be used as a backup or additional evaluation source.

The backup dataset is:

```text
michaelozon/candidate-matching-synthetic
```

This dataset is useful because it contains structured resumes, job postings, skills, and ground-truth matching records. It can help when we need clean labels for evaluation.

### Manual Evaluation Set

Because real public datasets do not always provide perfect human-labeled suitability judgments, the project will also create a small manual evaluation set.

Manual set size:

```text
10-20 resume-job pairs
```

Each pair will be labeled as:

- Strong match
- Partial match
- Weak match

This manual set will be used to check whether the system's score and HR explanation are reasonable.

### Final Dataset Strategy

```text
Main dataset:
cnamuangtoun/resume-job-description-fit

Additional real job postings:
1.3M LinkedIn Jobs & Skills (2024)

Resume source:
Public resume text from selected resume-job datasets

Skill dictionary:
Custom skills.json, optionally expanded with ESCO

Secondary dataset:
batuhanmtl/job_resume_fit

Backup dataset:
michaelozon/candidate-matching-synthetic

Manual evaluation:
10-20 manually labeled resume-job pairs
```

## 27. Summary

This project will be an explainable NLP-supported resume-job matching system for human resources.

The system will calculate semantic similarity between a resume and a job posting using an encoder-only Sentence Transformer model. It will also use dictionary-based skill extraction to show which skills match and which skills are missing.

The most important part of the project is explainability. The user should not only see a score. They should also understand why the score was produced, which areas are strong, and which areas need attention.
