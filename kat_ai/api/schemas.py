from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    language: str = Field(default="vi", pattern="^(vi|en|auto)$")
    task: str = Field(default="auto", max_length=40)
    max_new_tokens: int = Field(default=100, ge=1, le=256)
    temperature: float = Field(default=0.8, gt=0, le=1.5)

class ChatResponse(BaseModel):
    success: bool
    model: str
    task: str
    language: str
    response: str
    blocked: bool
    reason: str | None = None
    skill: str | None = None
    tool_result: dict | None = None

class QuizRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=120)
    count: int = Field(default=5, ge=1, le=20)
    difficulty: str = Field(default="intermediate", pattern="^(beginner|intermediate|advanced)$")
