import gradio as gr

# Local Imports - now importing from the DSPy version
from llm_chat_dspy import chat

if __name__ == "__main__":
    gr.ChatInterface(fn=chat, type="messages").launch()
