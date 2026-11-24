"""KatWork quiz skill for KatLearn's existing PHÒNG HỌC AI tab."""
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class Option:
    value: str
    label: str

# Small authored starter bank. Unknown topics fail closed; no fabricated questions.
BANK = {
 "present perfect": [
  ("She _____ here since 2020.", ["lives","lived","has lived","is living"], "C", "Use the present perfect with since for a situation continuing to now."),
  ("I _____ my homework yet.", ["haven't finished","didn't finish","don't finish","am not finishing"], "A", "Use haven't finished with yet for an unfinished action up to now."),
  ("_____ you ever visited London?", ["Did","Have","Do","Are"], "B", "Use Have to form a present perfect question."),
  ("They have _____ this film twice.", ["see","saw","seeing","seen"], "D", "The present perfect uses have/has plus the past participle."),
  ("Tom has worked here _____ five years.", ["since","for","from","during"], "B", "Use for with a duration."),
  ("We _____ each other for a long time.", ["know","knew","have known","are knowing"], "C", "Use the present perfect for a state continuing from the past."),
  ("My parents have _____ to Japan twice.", ["be","been","being","went"], "B", "The past participle of be is been."),
  ("She has not called me _____ Monday.", ["for","since","during","until"], "B", "Use since with a starting point."),
  ("How long _____ you lived here?", ["did","do","have","are"], "C", "Use have in a present perfect How long question."),
  ("He _____ just arrived.", ["have","has","is","was"], "B", "Use has with the third-person singular subject he.")
 ],
 "simple past": [
  ("We _____ to Da Nang last summer.", ["go","gone","went","going"], "C", "Use the past simple for a completed action at a finished time."),
  ("She _____ a new book yesterday.", ["buy","buys","bought","buying"], "C", "Yesterday signals the past simple; buy becomes bought."),
  ("_____ they watch the match last night?", ["Do","Did","Have","Are"], "B", "Use Did to form a past simple question."),
  ("I didn't _____ the answer.", ["knew","known","know","knowing"], "C", "After didn't, use the base form."),
  ("He _____ at home two hours ago.", ["is","was","were","be"], "B", "Use was with he in the past.")
 ],
 "articles": [
  ("She is _____ honest student.", ["a","an","the","no article"], "B", "Use an before a vowel sound; honest begins with a silent h."),
  ("I saw _____ moon last night.", ["a","an","the","no article"], "C", "Use the for a unique object in this context."),
  ("He wants to be _____ engineer.", ["a","an","the","no article"], "B", "Engineer begins with a vowel sound, so use an.")
 ]
}

def validate_question(q):
    if not isinstance(q, dict) or not isinstance(q.get("question"), str) or not q["question"].strip():
        return False
    options, answers = q.get("options"), q.get("correct_values")
    if not isinstance(options, list) or not 3 <= len(options) <= 6:
        return False
    values = [o.get("value") for o in options if isinstance(o, dict)]
    labels = [o.get("label") for o in options if isinstance(o, dict)]
    return (len(values) == len(options) and len(set(values)) == len(values)
            and all(isinstance(x, str) and x.strip() for x in labels)
            and isinstance(answers, list) and len(answers) == 1 and answers[0] in values
            and isinstance(q.get("explanation"), str) and bool(q["explanation"].strip()))

class KatWork:
    def create_quiz(self, topic: str, count: int = 5, difficulty: str = "intermediate"):
        if not 1 <= count <= 20:
            raise ValueError("count_must_be_between_1_and_20")
        if difficulty not in {"beginner","intermediate","advanced"}:
            raise ValueError("unsupported_difficulty")
        key = "simple past" if topic.casefold().strip() in {"past simple","past simple tense"} else topic.casefold().strip()
        rows = BANK.get(key)
        if rows is None:
            raise LookupError("topic_not_in_starter_bank_native_generator_not_configured")
        if count > len(rows):
            raise LookupError(f"only_{len(rows)}_distinct_questions_available_for_topic")
        questions = []
        for i, (stem, labels, answer, explanation) in enumerate(rows[:count], 1):
            q = {
                "id": f"katwork-{key.replace(' ','-')}-{i:03d}",
                "type": "single_select", "topic": topic, "difficulty": difficulty,
                "question": stem,
                "options": [asdict(Option(chr(65+j), label)) for j,label in enumerate(labels)],
                "correct_values": [answer], "explanation": explanation,
                "validation_status": "schema_validated_template"
            }
            if not validate_question(q):
                raise RuntimeError("built_in_question_failed_validation")
            questions.append(q)
        return {"skill":"KatWork","surface":"PHÒNG HỌC AI","topic":topic,
                "difficulty":difficulty,"question_count":len(questions),
                "questions":questions,"source":"authored_starter_bank","status":"ready"}
