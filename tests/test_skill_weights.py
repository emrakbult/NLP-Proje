from src.skill_weights import get_skill_weight, has_role_specific_skill, total_skill_weight


def test_technical_skills_weight_more_than_general_skills() -> None:
    assert get_skill_weight("Python") > get_skill_weight("Communication")
    assert get_skill_weight("AWS") > get_skill_weight("Leadership")
    assert get_skill_weight("Excel") < get_skill_weight("Python")


def test_total_skill_weight() -> None:
    assert total_skill_weight(["Python", "AWS"]) > total_skill_weight(["Communication", "Leadership"])


def test_has_role_specific_skill() -> None:
    assert has_role_specific_skill(["Communication", "Python"])
    assert has_role_specific_skill(["Data Analysis", "Statistics"])
    assert not has_role_specific_skill(["Communication", "Leadership"])
    assert not has_role_specific_skill(["Communication", "Excel"])
