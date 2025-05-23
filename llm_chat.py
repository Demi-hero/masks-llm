
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage

llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0,
)






system_message = [(
        "system",
        """
        You are a games master for a superhero RPG called masks. It is a game that uses 2d6 as well as a character stat for resolution.
        As a game it takes inspiration from things like young justice and young avengers. Your aim is to run a scene for a player from start to finish asking them questions based off of the prinicples of the game.
        """,
    )]

history = []

def chat(message:str, history:list, system_message=system_message):

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



