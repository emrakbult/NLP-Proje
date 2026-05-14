from __future__ import annotations


TECHNICAL_SKILLS = {
    ".NET",
    "ASP.NET",
    "AWS",
    "Airflow",
    "Angular",
    "Ansible",
    "Azure",
    "BERT",
    "Bash",
    "C",
    "C#",
    "C++",
    "CSS",
    "CI/CD",
    "Computer Vision",
    "Data Engineering",
    "Deep Learning",
    "Django",
    "Docker",
    "Elasticsearch",
    "Express.js",
    "FastAPI",
    "Flask",
    "Git",
    "GitHub Actions",
    "Go",
    "Google Cloud",
    "GraphQL",
    "HTML",
    "Hadoop",
    "Hugging Face",
    "Java",
    "JavaScript",
    "Jenkins",
    "Kafka",
    "Keras",
    "Kotlin",
    "Kubernetes",
    "Linux",
    "Machine Learning",
    "Microservices",
    "Microsoft SQL Server",
    "MongoDB",
    "MySQL",
    "Natural Language Processing",
    "NoSQL",
    "Node.js",
    "NumPy",
    "Oracle Database",
    "PHP",
    "Pandas",
    "PostgreSQL",
    "Power BI",
    "PyTorch",
    "Python",
    "R",
    "REST API",
    "React",
    "Redis",
    "Ruby",
    "SQL",
    "Scala",
    "Scikit-learn",
    "Spark",
    "Spring Boot",
    "Swift",
    "Tableau",
    "TensorFlow",
    "Terraform",
    "Transformer Models",
    "TypeScript",
    "Vue.js",
}

ANALYTICAL_SKILLS = {
    "Data Analysis",
    "Data Science",
    "Statistics",
}

BASIC_TOOL_SKILLS = {
    "Excel",
}

HR_DOMAIN_SKILLS = {
    "Employee Relations",
    "HRIS",
    "Performance Management",
    "Recruitment",
    "Training and Development",
}

GENERAL_SKILLS = {
    "Agile",
    "Communication",
    "Critical Thinking",
    "Customer Service",
    "Leadership",
    "Problem Solving",
    "Project Management",
    "Teamwork",
}

DEFAULT_SKILL_WEIGHT = 1.0
TECHNICAL_SKILL_WEIGHT = 1.4
ANALYTICAL_SKILL_WEIGHT = 1.1
BASIC_TOOL_SKILL_WEIGHT = 0.6
HR_DOMAIN_SKILL_WEIGHT = 1.0
GENERAL_SKILL_WEIGHT = 0.35
GENERAL_ONLY_SKILL_SCORE_CAP = 35.0


def get_skill_weight(skill: str) -> float:
    """Return the contribution weight for a canonical skill name."""

    if skill in TECHNICAL_SKILLS:
        return TECHNICAL_SKILL_WEIGHT
    if skill in ANALYTICAL_SKILLS:
        return ANALYTICAL_SKILL_WEIGHT
    if skill in BASIC_TOOL_SKILLS:
        return BASIC_TOOL_SKILL_WEIGHT
    if skill in HR_DOMAIN_SKILLS:
        return HR_DOMAIN_SKILL_WEIGHT
    if skill in GENERAL_SKILLS:
        return GENERAL_SKILL_WEIGHT
    return DEFAULT_SKILL_WEIGHT


def total_skill_weight(skills: list[str]) -> float:
    """Return total weight for a list of canonical skills."""

    return sum(get_skill_weight(skill) for skill in skills)


def has_role_specific_skill(skills: list[str]) -> bool:
    """Return whether a skill list contains technical, analytical, or domain-specific skills."""

    return any(skill in TECHNICAL_SKILLS or skill in ANALYTICAL_SKILLS or skill in HR_DOMAIN_SKILLS for skill in skills)
