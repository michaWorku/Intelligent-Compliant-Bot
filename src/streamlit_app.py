import streamlit as st
import os
import sys
import pandas as pd
from typing import List, Dict, Union

# --- Path Configuration for app.py being inside 'src/' ---
# Get the directory where app.py is located (e.g., .../your_project_root/src/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the project root directory (one level up from current_script_dir)
project_root = os.path.join(current_script_dir, '..')

# Add the project root to Python's path so modules in 'src' can be imported easily
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Now, import rag_pipeline assuming it's in src/rag_pipeline.py
    from src.rag_pipeline import Retriever, Generator, RAGPipeline
    print("Successfully imported RAG pipeline components.")
except ImportError as e:
    st.error(f"Error importing RAG pipeline components: {e}")
    st.error("Please ensure 'rag_pipeline.py' is in the 'src/' directory within your project structure.")
    st.stop() # Stop the app if core components can't be loaded


# --- Configuration with Corrected Relative Paths ---
# These paths are now relative to the 'project_root' because app.py is in 'src/'
FAISS_INDEX_PATH = os.path.join(project_root, 'vector_store', 'faiss_index', 'faiss_index.bin')
FAISS_METADATA_PATH = os.path.join(project_root, 'vector_store', 'faiss_index', 'faiss_metadata.csv')
CHROMADB_PATH = os.path.join(project_root, 'vector_store', 'chroma_db')
CHROMADB_COLLECTION_NAME = 'complaint_chunks'

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small" # Consider larger models if deploying with sufficient resources
TOP_K_RETRIEVAL = 5

# --- Global RAG Pipeline Initialization ---
# Use st.cache_resource to initialize the RAG pipeline only once
@st.cache_resource
def initialize_rag_pipeline():
    print("Initializing RAG components for Streamlit app...")
    try:
        rag_retriever = Retriever(
            embedding_model_name=EMBEDDING_MODEL_NAME,
            faiss_index_path=FAISS_INDEX_PATH,
            faiss_metadata_path=FAISS_METADATA_PATH,
            chromadb_path=CHROMADB_PATH,
            chromadb_collection_name=CHROMADB_COLLECTION_NAME
        )
        rag_generator = Generator(model_name=LLM_MODEL_NAME)
        rag_pipeline_instance = RAGPipeline(rag_retriever, rag_generator)
        print("RAG Pipeline initialized successfully for Streamlit app.")
        return rag_pipeline_instance
    except Exception as e:
        st.error(f"Failed to initialize RAG Pipeline: {e}")
        st.error("Please ensure Task 2 was completed and vector stores are correctly saved and accessible.")
        return None

rag_pipeline = initialize_rag_pipeline()
RAG_INITIALIZED = (rag_pipeline is not None)

# --- Helper Functions ---
def format_sources_for_display(sources: List[Dict]) -> str:
    """Formats retrieved sources into a readable string for Streamlit Markdown."""
    if not sources:
        return "No specific sources retrieved for this query."

    formatted_text = ""
    for i, source in enumerate(sources):
        # Limit text snippet length for display clarity
        snippet = source['text'][:250] + '...' if len(source['text']) > 250 else source['text']
        formatted_text += (
            f"**{i+1}. Product:** {source.get('product', 'N/A')}\n"
            f"   **Complaint ID:** {source.get('original_id', 'N/A')}\n"
            f"   **Snippet:** \"{snippet}\"\n\n"
        )
    return formatted_text

# --- Streamlit UI ---
st.set_page_config(page_title="CrediTrust Complaint Chatbot", layout="centered")

st.title("CrediTrust Complaint Analysis Chatbot")
st.markdown("Ask questions about customer complaints related to financial products (Credit Card, Personal Loan, BNPL, Savings Account, Money Transfer).")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = ""

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for new message
if prompt := st.chat_input("Your question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if not RAG_INITIALIZED:
                ai_response = "The RAG system is not initialized. Please check server logs for errors."
                st.session_state.sources = ""
            else:
                # Get vector store choice from radio button
                # This assumes the radio button is rendered before the prompt is submitted
                # For robustness, you might store this in session_state as well.
                vector_store_choice = st.session_state.get('vector_store_choice', 'faiss') # Default to faiss

                result = rag_pipeline.run(prompt, k=TOP_K_RETRIEVAL, vector_store_type=vector_store_choice)
                ai_response = result['answer']
                st.session_state.sources = format_sources_for_display(result['sources'])
            
            st.markdown(ai_response)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    
    # Vector Store Selection
    st.session_state.vector_store_choice = st.radio(
        "Choose Vector Store for Retrieval:",
        ("faiss", "chromadb"),
        key="vector_store_radio", # Unique key for the radio button
        help="Select which vector database to use for retrieving relevant complaint chunks."
    )

    # Clear Chat Button
    if st.button("Clear Chat", help="Clear the conversation history and retrieved sources."):
        st.session_state.messages = []
        st.session_state.sources = ""
        st.experimental_rerun() # Rerun the app to clear display

# Display sources in an expander below the chat
if st.session_state.sources:
    with st.expander("Show Retrieved Sources"):
        st.markdown(st.session_state.sources)