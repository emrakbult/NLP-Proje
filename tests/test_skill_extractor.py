from src.skill_extractor import compare_skills, extract_and_compare, extract_skills, load_skills


def test_extract_skills_normalizes_aliases() -> None:
    skills = load_skills()
    text = "Built RESTful services with Postgres and Amazon Web Services."

    extracted = extract_skills(text, skills)

    assert "REST API" in extracted
    assert "PostgreSQL" in extracted
    assert "AWS" in extracted


def test_extract_skills_supports_turkish_aliases() -> None:
    skills = load_skills()
    text = (
        "Python programlama, PostgreSQL veritabanı, REST API uç noktaları, "
        "veri analizi, çevik yazılım, iletişim ve takım çalışması deneyimi."
    )

    extracted = extract_skills(text, skills)

    assert "Python" in extracted
    assert "PostgreSQL" in extracted
    assert "REST API" in extracted
    assert "Data Analysis" in extracted
    assert "Agile" in extracted
    assert "Communication" in extracted
    assert "Teamwork" in extracted


def test_extract_skills_preserves_technical_terms() -> None:
    skills = load_skills()
    text = "Experience with C++, C#, .NET, Node.js, and React."

    extracted = extract_skills(text, skills)

    assert "C++" in extracted
    assert "C#" in extracted
    assert ".NET" in extracted
    assert "Node.js" in extracted
    assert "React" in extracted


def test_extract_skills_does_not_match_partial_words() -> None:
    skills = load_skills()
    text = "The candidate used JavaScript for frontend work."

    extracted = extract_skills(text, skills)

    assert "JavaScript" in extracted
    assert "Java" not in extracted


def test_extract_skills_does_not_auto_match_single_letter_skill_names() -> None:
    skills = load_skills()
    text = "REST APIs, CI/CD workflows, C++, and C# are listed, but R is not named as a skill."

    extracted = extract_skills(text, skills)

    assert "REST API" in extracted
    assert "CI/CD" in extracted
    assert "C++" in extracted
    assert "C#" in extracted
    assert "C" not in extracted
    assert "R" not in extracted


def test_compare_skills_returns_matched_missing_and_extra() -> None:
    result = compare_skills(
        resume_skills=["Python", "SQL", "Git", "Pandas"],
        job_skills=["Python", "SQL", "Docker", "AWS"],
    )

    assert result["matched_skills"] == ["Python", "SQL"]
    assert result["missing_skills"] == ["AWS", "Docker"]
    assert result["extra_resume_skills"] == ["Git", "Pandas"]


def test_extract_and_compare_resume_against_job_description() -> None:
    skills = load_skills()
    resume = "Software developer with Python, SQL, Git, and REST API experience."
    job = "We need Python, SQL, Git, Docker, AWS, and RESTful services experience."

    result = extract_and_compare(resume, job, skills)

    assert result["matched_skills"] == ["Git", "Python", "REST API", "SQL"]
    assert result["missing_skills"] == ["AWS", "Docker"]
