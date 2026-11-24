import pytest
from kat_ai.skills.router import route_intent
from kat_ai.skills.katwork import KatWork, validate_question
from kat_ai.skills.katgames import summarize_progress

def test_router_katwork():
    assert route_intent("Tạo 10 câu trắc nghiệm về Present Perfect").skill=="KatWork"

def test_katwork_quiz_pack():
    pack=KatWork().create_quiz("Present Perfect",10)
    assert pack["surface"]=="PHÒNG HỌC AI" and len(pack["questions"])==10
    assert all(validate_question(q) for q in pack["questions"])

def test_katwork_fails_closed_for_unknown_topic():
    with pytest.raises(LookupError): KatWork().create_quiz("conditionals",5)

def test_katgames_read_only():
    result=summarize_progress({"sessions":[{"score":9,"accuracy":.9,"average_reaction_ms":420}]})
    assert result["best_score"]==9 and result["mutations_performed"] is False

def test_katgames_rejects_invalid_data():
    with pytest.raises(ValueError): summarize_progress({"sessions":"invalid"})
