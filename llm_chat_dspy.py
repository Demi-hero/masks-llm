import logging
from typing import List

import chromadb
import dspy
from sentence_transformers import SentenceTransformer

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

lm = dspy.LM(f'ollama_chat/{config.MODEL}',
             api_base='http://localhost:11434',
             api_key='')

# Configure DSPy to use our Ollama model
dspy.configure(lm=lm)


class RAGRetriever:
    def __init__(self, db_path="./masks_rag_db"):
        """Connect to your RAG database"""
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            # Get the first collection (assuming you have one main collection)
            collections = self.client.list_collections()
            if collections:
                self.collection = collections[0]
                print(f"Connected to existing collection: {self.collection.name}")
            else:
                raise Exception("No collections found in the database")

            # Initialize embedding model (you might need to match what you used originally)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        except Exception as e:
            print(f"Error connecting to RAG database: {e}")
            self.collection = None

    def search(self, query: str, n_results: int = 3) -> List[str]:
        """Search your existing RAG database"""
        if not self.collection:
            return []

        try:
            query_embedding = self.embedding_model.encode([query]).tolist()

            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=['documents']
            )

            # Return just the documents as a list of strings
            return results['documents'][0] if results['documents'] else []

        except Exception as e:
            print(f"Error searching RAG database: {e}")
            return []


# Define a DSPy signature for chat
class ChatSignature(dspy.Signature):
    """You are a helpful AI assistant. Respond to the user's message based on the conversation history."""

    retrieved_context = dspy.InputField(desc="Relevant information retrieved from the knowledge base")
    system_message = dspy.InputField(desc="System instructions for the assistant")
    history = dspy.InputField(desc="Previous conversation history")
    user_message = dspy.InputField(desc="Current user message")
    response = dspy.OutputField(desc="Assistant's response to the user")


# Create a DSPy module for chat
class ChatModule(dspy.Module):
    def __init__(self, rag_retriever: RAGRetriever):
        super().__init__()
        self.retriever = rag_retriever
        self.chat_predictor = dspy.ChainOfThought(ChatSignature)
        # Incase no context is gathered
        self.fallback_predictor = dspy.ChainOfThought("system_message, history, user_message -> response")

    def forward(self, user_message: str, history: list, system_message: str = config.SYS_PROMPT):
        # Format history as a string
        history_str = self._format_history(history)
        retrieved_docs = self.retriever.search(user_message, n_results=config.DOC_COUNT)

        if retrieved_docs:
            # Format the retrieved context
            context = "\n\n".join([f"Context {i + 1}: {doc}" for i, doc in enumerate(retrieved_docs)])
            logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")

            # Generate response with context
            result = self.chat_predictor(
                retrieved_context=context,
                system_message=system_message,
                history=history_str,
                user_message=user_message
            )

            return result.response
        else:
            # No relevant context found, use regular chat
            print("No relevant context found, using fallback")
            result = self.fallback_predictor(
                system_message=system_message,
                history=history_str,
                user_message=user_message
            )

            return result.response

    def _format_history(self, history):
        if not history:
            return "No previous conversation."

        formatted = []
        for item in history:
            if isinstance(item, dict):
                role = item.get('role', 'user')
                content = item.get('content', '')
                formatted.append(f"{role.capitalize()}: {content}")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                formatted.append(f"{item[0].capitalize()}: {item[1]}")

        return "\n".join(formatted)


# Initialize the chat module
# Initialize RAG components
rag_retriever = RAGRetriever(db_path="./masks_rag_db")
chat_module = ChatModule(rag_retriever)


def chat(message: str, history: list, system_message=config.SYS_PROMPT):
    """
    DSPy version of the chat function
    Note: DSPy doesn't have built-in streaming like LangChain,
    so this returns the complete response at once
    """
    # Handle both Gradio message format and tuple format
    converted_history = []

    if history:
        for item in history:
            if isinstance(item, dict):
                # Gradio format: {'role': 'user', 'content': 'message'}
                role = item.get('role', 'user')
                content = item.get('content', '')
                converted_history.append((role, content))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                # Tuple format: ('user', 'message')
                converted_history.append((item[0], item[1]))

    logger.info("History is: %s", converted_history)

    try:
        response = chat_module.forward(
            user_message=message,
            history=converted_history,
            system_message=system_message
        )

        print("DSPy response:")
        print(response)

        # Return as generator to maintain API compatibility
        yield response

    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print(error_msg)
        yield error_msg


# Alternative: Simple DSPy approach without custom module
def simple_chat(message: str, history: list, system_message=config.SYS_PROMPT):
    """Simplified DSPy chat function using basic predict"""

    # Format the conversation
    conversation = system_message + "\n\n"

    for role, msg in history:
        conversation += f"{role.capitalize()}: {msg}\n"

    conversation += f"User: {message}\nAssistant:"

    # Use DSPy's basic prediction
    response = lm.basic_request(conversation)

    print("History is:")
    print(history)
    print("Response:", response[0])

    yield response[0]


# Example usage with DSPy optimization capabilities
class OptimizedChatModule(dspy.Module):
    """
    This version can be optimized using DSPy's built-in optimizers
    """

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(ChatSignature)

    def forward(self, user_message, history, system_message=config.SYS_PROMPT):
        history_str = self._format_history(history)

        return self.generate(
            system_message=system_message,
            history=history_str,
            user_message=user_message
        ).response

    def _format_history(self, history):
        if not history:
            return "No previous conversation."

        formatted = []
        for role, message in history:
            formatted.append(f"{role.capitalize()}: {message}")

        return "\n".join(formatted)
