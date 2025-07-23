import gradio as gr

# Local Imports
from llm_chat_lang import chat

if __name__ == "__main__":
    gr.ChatInterface(fn=chat, type="messages").launch()
