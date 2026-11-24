"""Conservative English-learning scope gate; not a semantic classifier."""
from dataclasses import dataclass
import re
import unicodedata

REFUSAL = "Hiện tại, chúng tôi không hỗ trợ các môn học khác ngoài Tiếng Anh. Xin cảm ơn"
ENGLISH_SIGNALS = (
    "english", "translate", "translation", "grammar", "vocabulary", "vocab",
    "meaning of", "definition", "synonym", "antonym", "collocation", "essay",
    "paragraph", "writing", "proofread", "reading comprehension", "ielts",
    "pronunciation", "phrasal verb", "present perfect", "past simple", "simple past",
    "future simple", "sentence", "multiple choice", "quiz", "english quiz",
    "ngữ pháp", "từ vựng", "phát âm", "dịch câu", "dịch đoạn", "dịch sang",
    "trắc nghiệm tiếng anh", "ôn tập tiếng anh", "bài tập tiếng anh",
    "thì hiện tại", "thì quá khứ", "mệnh đề quan hệ", "game progress",
    "reflex game", "tiến độ game", "game phản xạ", "lịch sử chơi",
)
OUT_OF_SCOPE_SIGNALS = (
    "solve this equation", "calculate the answer", "physics homework",
    "chemistry homework", "biology homework", "history homework",
    "geography homework", "solve this math", "giải phương trình",
    "bài tập vật lý", "bài tập hóa học", "bài tập sinh học",
    "bài tập lịch sử", "bài tập địa lý", "giải bài toán",
)

@dataclass(frozen=True)
class ScopeResult:
    allowed: bool
    reason: str

def classify_scope(message: str) -> ScopeResult:
    text = unicodedata.normalize("NFKC", message or "").casefold()
    text = re.sub(r"\\s+", " ", text).strip()
    if not text:
        return ScopeResult(False, "empty")
    out_of_scope = any(term in text for term in OUT_OF_SCOPE_SIGNALS)
    translation = any(term in text for term in ("translate", "translation", "dịch câu", "dịch đoạn", "dịch sang"))
    if out_of_scope and not translation:
        return ScopeResult(False, "outside_english_learning")
    if any(term in text for term in ENGLISH_SIGNALS):
        return ScopeResult(True, "english_learning_signal")
    return ScopeResult(False, "unclear_or_outside_scope")
