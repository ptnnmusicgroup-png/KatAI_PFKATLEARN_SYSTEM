"""Deterministic intent router; it never grants permissions."""
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Route:
    skill: str
    reason: str

QUIZ_PATTERNS = (
    r"\\b(create|make|generate|build)\\b.{0,50}\\b(quiz|multiple[- ]choice|questions|test)\\b",
    r"\\b(quiz|multiple[- ]choice|questions|test)\\b.{0,50}\\b(about|on|for|with)\\b",
    r"(tạo|ra đề|soạn).{0,50}(câu hỏi|trắc nghiệm|đề kiểm tra|quiz)",
    r"(câu hỏi|trắc nghiệm|đề kiểm tra).{0,50}(về|chủ đề|unit)",
)
GAME_PATTERNS = (r"\\b(game|reflex|reaction time|game progress)\\b", r"(tiến độ game|game phản xạ|lịch sử chơi)")

def route_intent(message: str, requested_task: str = "auto") -> Route:
    task = (requested_task or "auto").casefold()
    text = re.sub(r"\\s+", " ", message.casefold()).strip()
    if task in {"katwork", "quiz", "quiz_generation"} or any(re.search(p, text) for p in QUIZ_PATTERNS):
        return Route("KatWork", "quiz_generation_intent")
    if task in {"katgames", "game_progress"} or any(re.search(p, text) for p in GAME_PATTERNS):
        return Route("KatGames", "game_progress_intent")
    return Route("KatAICore", "default_learning_assistance")
