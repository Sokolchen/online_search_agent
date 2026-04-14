# app/main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import json
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
# 加载环境变量
load_dotenv()

# 导入agent
from src.agent.agent import agent

app = FastAPI(title="Ken Masters Agent API")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# --- 请求/响应模型 ---
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None  # 用于区分不同会话


class ChatResponse(BaseModel):
    response: str


# --- 非流式调用（一次性返回最终答案）---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # 构建输入
    inputs = {"messages": [("user", req.message)]}

    # 配置（用于记忆）
    config = {}
    if req.thread_id:
        config["configurable"] = {"thread_id": req.thread_id}

    # 调用 agent（假设是异步的）
    result = await agent.ainvoke(inputs, config=config)

    # 提取最后一条 AI 消息
    messages = result.get("messages", [])
    ai_response = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            ai_response = msg.content
            break

    return ChatResponse(response=ai_response)


# --- 流式调用（实时返回 token）---
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    inputs = {"messages": [("user", req.message)]}
    config = {}
    if req.thread_id:
        config["configurable"] = {"thread_id": req.thread_id}

    async def event_generator():
        # 使用 astream_events 捕获 token 流
        async for event in agent.astream_events(inputs, config=config, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield f"data: {json.dumps({'content': content})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- 健康检查 ---
@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)