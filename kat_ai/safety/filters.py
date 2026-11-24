"""Layered deterministic request/output checks; not a complete safety solution."""
import re
import unicodedata

PATTERNS = (
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|messages?)",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?)",
    r"(reveal|print|show|repeat|expose)\s+(the\s+)?(system prompt|hidden instructions|developer message|api key|secret)",
    r"(act as|pretend to be)\s+(an?\s+)?(unrestricted|uncensored|jailbroken|no[- ]rules)\s+(ai|assistant|model)",
    r"bỏ qua\s+(toàn bộ\s+)?(chỉ dẫn|hướng dẫn|quy tắc)\s+(trước|ở trên)",
    r"tiết lộ\s+(system prompt|chỉ dẫn hệ thống|khóa api|mật khẩu)",
)

def normalize_security_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return re.sub(r"\s+", " ", text).strip()

def contains_prompt_injection(text: str) -> bool:
    normalized = normalize_security_text(text)
    return any(re.search(p, normalized, re.I) for p in PATTERNS)

def validate_tool_payload(payload, max_items=50, max_text=4000) -> bool:
    if not isinstance(payload, dict) or len(payload) > 32:
        return False
    for value in payload.values():
        if isinstance(value, str) and len(value) > max_text: return False
        if isinstance(value, list) and len(value) > max_items: return False
    return True

def output_is_acceptable(text, max_chars=12000):
    return isinstance(text, str) and bool(text.strip()) and len(text) <= max_chars
