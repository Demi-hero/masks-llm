import gradio as gr

# Local Imports
from llm_chat import chat



gr.ChatInterface(fn=chat, type="messages").launch()