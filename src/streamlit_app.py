import streamlit as st
import os
import sys
import pandas as pd
from typing import List, Dict, Union

# Import Hugging Face Hub client
from huggingface_hub import hf_hub_download, HfFileSystem

# --- Path Configuration for app.py being inside 'src/' ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_script_dir, '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.rag_pipeline import Retriever, Generator, RAGPipeline, CHROMADB_AVAILABLE
    print("Successfully imported RAG pipeline components.")
except ImportError as e:
    st.error(f"Error importing RAG pipeline components: {e}")
    st.error("Please ensure 'rag_pipeline.py' is in the 'src/' directory within your project structure.")
    st.stop()


# --- Configuration for Vector Store Paths and Hugging Face Hub ---
# Local paths where vector stores will be downloaded
LOCAL_VECTOR_STORE_DIR = "downloaded_vector_store" # A temporary directory in the container
LOCAL_FAISS_INDEX_DIR = os.path.join(LOCAL_VECTOR_STORE_DIR, 'faiss_index')
LOCAL_CHROMADB_DIR = os.path.join(LOCAL_VECTOR_STORE_DIR, 'chroma_db')

FAISS_INDEX_PATH = os.path.join(LOCAL_FAISS_INDEX_DIR, 'faiss_index.bin')
FAISS_METADATA_PATH = os.path.join(LOCAL_FAISS_INDEX_DIR, 'faiss_metadata.csv')
CHROMADB_PATH = LOCAL_CHROMADB_DIR # ChromaDB expects a directory path
CHROMADB_COLLECTION_NAME = 'complaint_chunks'

# Hugging Face Hub Repository details
HF_REPO_ID = "michaWorku/credittrust-rag" #
HF_REPO_TYPE = "dataset" # Or "model" if you chose that type

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"
TOP_K_RETRIEVAL = 5


# --- Function to Download Vector Store from Hugging Face Hub ---
@st.cache_resource
def download_vector_store_from_hf(repo_id: str, repo_type: str, local_dir: str):
    """
    Downloads vector store files from Hugging Face Hub to a local directory.
    """
    if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 0:
        print(f"Vector store already exists at {local_dir}. Skipping download.")
        return

    print(f"Downloading vector store from Hugging Face Hub '{repo_id}' to '{local_dir}'...")
    
    # Ensure local directories exist
    os.makedirs(os.path.join(local_dir, 'faiss_index'), exist_ok=True)
    os.makedirs(os.path.join(local_dir, 'chroma_db'), exist_ok=True)

    fs = HfFileSystem()
    
    # List all files in the 'vector_store' directory within the HF repo
    # This assumes your files are directly under the 'vector_store' folder in the repo
    # e.g., your_hf_username/credittrust-rag/vector_store/faiss_index/...
    
    # Adjust this prefix if your files are directly at the root of the HF dataset repo
    # If your files are directly in the root of the HF dataset, use "" as prefix.
    # If they are under a 'vector_store' folder, use "vector_store/"
    hf_folder_prefix = "vector_store/" # Adjust if your HF repo structure is different

    try:
        files_to_download = fs.ls(f"{repo_id}/{hf_folder_prefix}", detail=False)
        download_count = 0
        for hf_file_path in files_to_download:
            # Extract the relative path within the 'vector_store' folder
            relative_path = os.path.relpath(hf_file_path, f"{repo_id}/{hf_folder_prefix}")
            local_file_path = os.path.join(local_dir, relative_path)
            
            # Ensure subdirectory exists for the file
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            if not hf_file_path.endswith('/'): # Skip directories themselves
                print(f"Downloading {hf_file_path} to {local_file_path}...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=os.path.join(hf_folder_prefix, relative_path), # Path within the HF repo
                    repo_type=repo_type,
                    local_dir=local_dir, # Download to the base local_dir
                    local_dir_use_symlinks=False # Important for non-model files
                )
                download_count += 1
        
        if download_count > 0:
            print(f"Successfully downloaded {download_count} files from Hugging Face Hub.")
        else:
            st.warning(f"No files found in Hugging Face Hub repo '{repo_id}' under prefix '{hf_folder_prefix}'. "
                       f"Please check your repo ID and prefix. Vector stores might not be loaded.")

    except Exception as e:
        st.error(f"Failed to download files from Hugging Face Hub: {e}")
        st.error("Please ensure your HF_REPO_ID and HF_REPO_TYPE are correct and the files exist.")
        st.stop()


# --- Global RAG Pipeline Initialization ---
@st.cache_resource
def initialize_rag_pipeline():
    # First, download the vector store files
    download_vector_store_from_hf(HF_REPO_ID, HF_REPO_TYPE, LOCAL_VECTOR_STORE_DIR)

    print("Initializing RAG components for Streamlit app...")
    try:
        rag_retriever = Retriever(
            embedding_model_name=EMBEDDING_MODEL_NAME,
            faiss_index_path=FAISS_INDEX_PATH,
            faiss_metadata_path=FAISS_METADATA_PATH,
            chromadb_path=CHROMADB_PATH, # This is now the local downloaded path
            chromadb_collection_name=CHROMADB_COLLECTION_NAME
        )
        rag_generator = Generator(model_name=LLM_MODEL_NAME)
        rag_pipeline_instance = RAGPipeline(rag_retriever, rag_generator)
        print("RAG Pipeline initialized successfully for Streamlit app.")
        return rag_pipeline_instance
    except Exception as e:
        st.error(f"Failed to initialize RAG Pipeline: {e}")
        st.error("Please ensure vector stores are correctly downloaded and accessible.")
        return None

rag_pipeline = initialize_rag_pipeline()
RAG_INITIALIZED = (rag_pipeline is not None)

# --- Helper Functions (rest of the app.py remains the same) ---
def format_sources_for_display(sources: List[Dict]) -> str:
    """Formats retrieved sources into a readable string for Streamlit Markdown."""
    if not sources:
        return "No specific sources retrieved for this query."

    formatted_text = ""
    for i, source in enumerate(sources):
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

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if not RAG_INITIALIZED:
                ai_response = "The RAG system is not initialized. Please check server logs for errors."
                st.session_state.sources = ""
            else:
                vector_store_choice = st.session_state.get('vector_store_choice', 'faiss')

                if vector_store_choice == 'chromadb' and (not CHROMADB_AVAILABLE or rag_pipeline.retriever.chroma_collection is None):
                    ai_response = "ChromaDB is not available. Please select FAISS for retrieval."
                    st.session_state.sources = ""
                elif vector_store_choice == 'faiss' and (rag_pipeline.retriever.faiss_index is None or rag_pipeline.retriever.faiss_metadata is None):
                    ai_response = "FAISS is not available. Please check the vector store files."
                    st.session_state.sources = ""
                else:
                    result = rag_pipeline.run(prompt, k=TOP_K_RETRIEVAL, vector_store_type=vector_store_choice)
                    ai_response = result['answer']
                    st.session_state.sources = format_sources_for_display(result['sources'])
            
            st.markdown(ai_response)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

with st.sidebar:
    st.header("Settings")
    
    options = ["faiss"]
    if CHROMADB_AVAILABLE and rag_pipeline and rag_pipeline.retriever.chroma_collection:
        options.append("chromadb")
    
    st.session_state.vector_store_choice = st.radio(
        "Choose Vector Store for Retrieval:",
        options,
        index=0,
        key="vector_store_radio",
        help="Select which vector database to use for retrieving relevant complaint chunks."
    )
    
    if "chromadb" not in options:
        st.warning("ChromaDB is not available. This might be due to deployment environment limitations or missing files.")

    if st.button("Clear Chat", help="Clear the conversation history and retrieved sources."):
        st.session_state.messages = []
        st.session_state.sources = ""
        st.experimental_rerun()

if st.session_state.sources:
    with st.expander("Show Retrieved Sources"):
        st.markdown(st.session_state.sources)