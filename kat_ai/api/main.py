import os
import re
from functools import lru_cache
from fastapi import FastAPI, Header, HTTPException
from kat_ai import __version__
from kat_ai.api.schemas import ChatRequest, ChatResponse, QuizRequest
from kat_ai.safety.scope import classify_scope, REFUSAL
from kat_ai.safety.filters import contains_prompt_injection, output_is_acceptable
from kat_ai.inference.engine import InferenceEngine
from kat_ai.skills.router import route_intent
from kat_ai.skills.katwork import KatWork

app=FastAPI(title="KatAI Native API",version=__version__,description="Local-first English-learning AI with first-party skills.")
katwork=KatWork()

@lru_cache(maxsize=1)
def get_engine():
    checkpoint=os.getenv("KATAI_CHECKPOINT","checkpoints/katai.pt")
    tokenizer=os.getenv("KATAI_TOKENIZER","artifacts/tokenizer/tokenizer.model")
    if not os.path.isfile(checkpoint) or not os.path.isfile(tokenizer): return None
    return InferenceEngine(checkpoint,tokenizer)

def verify_auth(authorization):
    expected=os.getenv("KATAI_API_KEY")
    if expected and authorization!=f"Bearer {expected}":
        raise HTTPException(status_code=401,detail="Invalid or missing API key.")

def extract_quiz_topic(message):
    for topic in ("present perfect","simple past","past simple","articles"):
        if topic in message.casefold(): return "simple past" if topic=="past simple" else topic
    match=re.search(r"(?:about|on|for|về|chủ đề)\s+(.+?)(?:[.!?]|$)",message,re.I)
    return match.group(1).strip().strip("\"' ") if match else ""

def make_quiz(topic,count=5,difficulty="intermediate"):
    if not topic: raise HTTPException(status_code=422,detail="Please specify the English topic.")
    try: return katwork.create_quiz(topic,count,difficulty)
    except LookupError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc

@app.get("/health")
def health():
    return {"status":"ok","model_loaded":get_engine() is not None,"version":__version__}

@app.get("/v1/model/info")
def model_info(authorization: str | None=Header(default=None)):
    verify_auth(authorization); engine=get_engine()
    if engine is None: return {"loaded":False,"model":"katai-native","version":__version__,"skills":["KatWork","KatGames"]}
    return {"loaded":True,"model":"katai-native","version":__version__,"architecture":"decoder-only-transformer",
        "origin":"trained-from-scratch","languages":["en","vi"],"training_step":engine.step,
        "parameters":sum(p.numel() for p in engine.model.parameters()),"device":str(engine.device),
        "skills":["KatWork","KatGames"]}

@app.post("/v1/katwork/quiz")
def create_katwork_quiz(request: QuizRequest,authorization: str | None=Header(default=None)):
    verify_auth(authorization)
    return make_quiz(request.topic,request.count,request.difficulty)

@app.post("/v1/chat",response_model=ChatResponse)
def chat(request: ChatRequest,authorization: str | None=Header(default=None)):
    verify_auth(authorization)
    if contains_prompt_injection(request.message):
        return ChatResponse(success=True,model="katai-native",task=request.task,language=request.language,
            response="Mình có thể hỗ trợ các nhiệm vụ học tiếng Anh trong phạm vi KatAI.",
            blocked=True,reason="prompt_injection_pattern")
    scope=classify_scope(request.message)
    if not scope.allowed:
        return ChatResponse(success=True,model="katai-native",task=request.task,language=request.language,
            response=REFUSAL,blocked=True,reason=scope.reason)
    route=route_intent(request.message,request.task)
    if route.skill=="KatWork":
        topic=extract_quiz_topic(request.message)
        found=re.search(r"\b(\d{1,2})\s*(?:câu|questions?)?\b",request.message.casefold())
        count=min(20,max(1,int(found.group(1)))) if found else 5
        quiz=make_quiz(topic,count)
        return ChatResponse(success=True,model="katai-native",task="quiz_generation",language=request.language,
            response=f"Mình đã tạo {quiz['question_count']} câu hỏi về {topic} cho PHÒNG HỌC AI.",
            blocked=False,skill="KatWork",tool_result=quiz)
    if route.skill=="KatGames":
        return ChatResponse(success=True,model="katai-native",task="game_progress",language=request.language,
            response="KatGames đã nhận diện yêu cầu. Tiến độ chỉ được đọc qua API KatLearn đã xác thực; adapter chưa được cấu hình.",
            blocked=False,skill="KatGames",reason="katlearn_progress_adapter_not_configured")
    engine=get_engine()
    if engine is None: raise HTTPException(status_code=503,detail="No trained native checkpoint/tokenizer configured.")
    prompt=f"<task>{request.task}</task><language>{request.language}</language><user>{request.message}</user><assistant>"
    result=engine.generate(prompt,max_new_tokens=request.max_new_tokens,temperature=request.temperature)
    if not output_is_acceptable(result): raise HTTPException(status_code=500,detail="Generated output failed validation.")
    return ChatResponse(success=True,model="katai-native",task=request.task,language=request.language,
        response=result,blocked=False,skill="KatAICore")
