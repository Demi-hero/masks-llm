
from langchain_ollama import ChatOllama

import config

llm = ChatOllama(
    model=config.MODEL,
    temperature=config.TEMP,
)


def chat(message: str, history: list, system_message=config.SYS_PROMPT):
    messages = system_message + history + [("user", message)]
    print("History is:")
    print(history)
    print("And messages is:")
    print(messages)

    ai_msg = llm.stream(messages)
    response = ""
    for chunk in ai_msg:
        response += chunk.content or ''
        yield response