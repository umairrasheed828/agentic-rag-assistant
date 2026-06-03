import asyncio
from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.agent import search_papers  # reuse the in-process tool
from src.agent.llm import get_llm
from langgraph.errors import GraphRecursionError

SYSTEM = SystemMessage(
    "You are a research assistant. Use search_papers for questions about paper content, "
    "and corpus_overview for questions about the knowledge base itself. Cite paper titles, "
    "and say you don't know if nothing relevant comes back. Reply directly to greetings."
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


async def build_agent():
    client = MultiServerMCPClient(
        {
            "corpus": {
                "command": "uv",
                "args": ["run", "python", "-m", "src.mcp_server"],
                "transport": "stdio",
            }
        }
    )
    mcp_tools = await client.get_tools()  # corpus_overview, loaded over MCP
    tools = [search_papers, *mcp_tools]  # local tool + MCP tool, combined
    llm_with_tools = get_llm().bind_tools(tools)

    async def agent_node(state: AgentState) -> dict:
        response = await llm_with_tools.ainvoke([SYSTEM] + state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


MAX_INPUT_CHARS = 2000


async def main(question: str) -> None:
    # Input guardrail: reject empty or oversized input before spending any tokens
    question = question.strip()
    if not question:
        print("Please provide a non-empty question.")
        return
    if len(question) > MAX_INPUT_CHARS:
        print(f"Question too long (max {MAX_INPUT_CHARS} characters).")
        return

    agent = await build_agent()
    try:
        # Loop guardrail: cap how many agent<->tools cycles can run
        result = await agent.ainvoke(
            {"messages": [("human", question)]},
            config={"recursion_limit": 8},
        )
        print(result["messages"][-1].content)
    except GraphRecursionError:
        print(
            "The agent ran too many steps without finishing. Try rephrasing your question."
        )


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "How many papers are in your knowledge base?"
    asyncio.run(main(q))
