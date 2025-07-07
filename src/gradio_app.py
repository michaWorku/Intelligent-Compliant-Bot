import gradio as gr
import os
import sys
import pandas as pd
from typing import List, Dict, Union

# --- Path Configuration ---
# Get the directory where app.py is located (e.g., .../your_project_root/src/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the project root directory (one level up from current_script_dir)
project_root = os.path.join(current_script_dir, '..')

# Add the project root to Python's path so modules in 'src' can be imported easily
# and other top-level directories like 'vector_store' are accessible.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# If rag_pipeline.py is in 'src/' (which it should be), this will help import it.
# We've already added project_root, so src/ should be discoverable.
# sys.path.insert(0, current_script_dir) # This might be redundant but ensures src is on path


try:
    # Now, import rag_pipeline assuming it's in a discoverable path like src/
    from src.rag_pipeline import Retriever, Generator, RAGPipeline
    print("Successfully imported RAG pipeline components.")
except ImportError as e:
    print(f"Error importing RAG pipeline components: {e}")
    print("Please ensure 'rag_pipeline.py' is in the 'src/' directory within your project structure.")
    sys.exit(1)


# --- Configuration with Corrected Relative Paths ---
# These paths are now relative to the 'project_root' because app.py is in 'src/'
FAISS_INDEX_PATH = os.path.join(project_root, 'vector_store', 'faiss_index', 'faiss_index.bin')
FAISS_METADATA_PATH = os.path.join(project_root, 'vector_store', 'faiss_index', 'faiss_metadata.csv')
CHROMADB_PATH = os.path.join(project_root, 'vector_store', 'chroma_db')
CHROMADB_COLLECTION_NAME = 'complaint_chunks'

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"
TOP_K_RETRIEVAL = 5

# --- Global RAG Pipeline Initialization ---
print("Initializing RAG components for Gradio app...")
try:
    rag_retriever = Retriever(
        embedding_model_name=EMBEDDING_MODEL_NAME,
        faiss_index_path=FAISS_INDEX_PATH,
        faiss_metadata_path=FAISS_METADATA_PATH,
        chromadb_path=CHROMADB_PATH,
        chromadb_collection_name=CHROMADB_COLLECTION_NAME
    )
    rag_generator = Generator(model_name=LLM_MODEL_NAME)
    rag_pipeline = RAGPipeline(rag_retriever, rag_generator)
    print("RAG Pipeline initialized successfully for Gradio app.")
    RAG_INITIALIZED = True
except Exception as e:
    print(f"Failed to initialize RAG Pipeline: {e}")
    print("Gradio app will run in a limited capacity or might not start.")
    rag_pipeline = None
    RAG_INITIALIZED = False

# --- Gradio Interface Logic ---

def format_sources_for_display(sources: List[Dict]) -> str:
    """Formats retrieved sources into a readable string for Gradio Markdown."""
    if not sources:
        return "No specific sources retrieved for this query."

    formatted_text = "#### Retrieved Sources:\n"
    for i, source in enumerate(sources):
        # Limit text snippet length for display clarity
        snippet = source['text'][:250] + '...' if len(source['text']) > 250 else source['text']
        formatted_text += (
            f"**{i+1}. Product:** {source.get('product', 'N/A')}\n"
            f"   **Complaint ID:** {source.get('original_id', 'N/A')}\n"
            f"   **Snippet:** \"{snippet}\"\n\n"
        )
    return formatted_text

def respond(message: str, chat_history: List[List[str]], vector_store_choice: str) -> (List[List[str]], str):
    """
    Handles user input, runs the RAG pipeline, and formats the output for Gradio.
    
    Args:
        message (str): The user's input question.
        chat_history (List[List[str]]): Gradio's chat history.
        vector_store_choice (str): The chosen vector store ('faiss' or 'chromadb').

    Returns:
        tuple: Updated chat history and the markdown-formatted sources.
    """
    if not RAG_INITIALIZED:
        ai_response = "The RAG system is not initialized. Please check server logs for errors."
        chat_history.append([message, ai_response])
        return chat_history, ""

    print(f"User query: {message} | Vector Store: {vector_store_choice}")
    
    # Run the RAG pipeline
    result = rag_pipeline.run(message, k=TOP_K_RETRIEVAL, vector_store_type=vector_store_choice)
    ai_answer = result['answer']
    retrieved_sources = result['sources']

    # Add AI response to chat history
    chat_history.append([message, ai_answer])

    # Format sources for display in a separate markdown component
    sources_markdown = format_sources_for_display(retrieved_sources)

    return chat_history, sources_markdown

def clear_chat():
    """Clears the chat history and sources display."""
    return [], ""

# --- Gradio Blocks Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # CrediTrust Complaint Analysis Chatbot
        Ask questions about customer complaints related to financial products (Credit Card, Personal Loan, BNPL, Savings Account, Money Transfer).
        """
    )

    chatbot = gr.Chatbot(
        label="Conversation History",
        height=400,
        avatar_images=(None, os.path.join(current_script_dir, 'static', 'bot_avatar.png')) # Optional: Add a bot avatar
    )

    with gr.Row():
        msg = gr.Textbox(
            label="Your Question",
            placeholder="e.g., What are common complaints about credit card billing?",
            scale=4
        )
        submit_btn = gr.Button("Ask", scale=1, variant="primary")
    
    with gr.Row():
        clear_btn = gr.Button("Clear Chat")
        vector_store_radio = gr.Radio(
            ["faiss", "chromadb"], 
            value="faiss", 
            label="Choose Vector Store for Retrieval", 
            interactive=True
        )

    sources_display = gr.Markdown(
        value="#### Retrieved Sources:", 
        label="Retrieved Source Documents",
        elem_id="sources_display_area" # Optional: for custom CSS if needed
    )

    # Event handlers
    submit_btn.click(
        fn=respond,
        inputs=[msg, chatbot, vector_store_radio],
        outputs=[chatbot, sources_display],
        queue=False
    ).then(
        fn=lambda: gr.update(value=""), # Clear the input box after submission
        inputs=None,
        outputs=[msg]
    )

    msg.submit( # Allow pressing Enter to submit
        fn=respond,
        inputs=[msg, chatbot, vector_store_radio],
        outputs=[chatbot, sources_display],
        queue=False
    ).then(
        fn=lambda: gr.update(value=""),
        inputs=None,
        outputs=[msg]
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[chatbot, sources_display],
        queue=False
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    # Ensure a 'static' directory exists for avatars if used
    static_dir = os.path.join(current_script_dir, 'static')
    os.makedirs(static_dir, exist_ok=True)
    # Create a dummy bot avatar if it doesn't exist for demonstration
    dummy_avatar_path = os.path.join(static_dir, 'bot_avatar.png')
    if not os.path.exists(dummy_avatar_path):
        from PIL import Image
        img = Image.new('RGB', (60, 60), color = (73, 109, 137))
        img.save(dummy_avatar_path)
    
    demo.launch(share=True, debug=True)