from langchain_openai import ChatOpenAI

from src.config import settings


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key,
        temperature=0,
    )


if __name__ == "__main__":
    llm = get_llm()
    response = llm.invoke("In one sentence, what is retrieval-augmented generation?")
    print(response.content)
