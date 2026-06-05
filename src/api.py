from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agent.agent import agent_graph

app = FastAPI(title="Agentic RAG Assistant")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:  # input guardrail, carried over from Phase 3
        raise HTTPException(status_code=400, detail="question must not be empty")
    result = agent_graph.invoke({"messages": [("human", question)]})
    return AskResponse(answer=str(result["messages"][-1].content))
