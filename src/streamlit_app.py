import streamlit as st
import os
import sys
import pandas as pd
from typing import List, Dict, Union

# Import Hugging Face Hub client
from huggingface_hub import hf_hub_download, list_repo_files

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
LOCAL_VECTOR_STORE_DIR = "downloaded_vector_store" # A temporary base directory in the container

# These paths now include the 'vector_store' subdirectory that is created during download.
LOCAL_FAISS_INDEX_DIR = os.path.join(LOCAL_VECTOR_STORE_DIR, 'vector_store', 'faiss_index')
LOCAL_CHROMADB_DIR = os.path.join(LOCAL_VECTOR_STORE_DIR, 'vector_store', 'chroma_db')

FAISS_INDEX_PATH = os.path.join(LOCAL_FAISS_INDEX_DIR, 'faiss_index.bin')
FAISS_METADATA_PATH = os.path.join(LOCAL_FAISS_INDEX_DIR, 'faiss_metadata.csv')
CHROMADB_PATH = LOCAL_CHROMADB_DIR # ChromaDB expects a directory path
CHROMADB_COLLECTION_NAME = 'complaint_chunks'

# Hugging Face Hub Repository details
HF_REPO_ID = "michaWorku/credittrust-rag"
HF_REPO_TYPE = "dataset"

# This is the prefix for the folder *within* the Hugging Face dataset
# where your vector store files are located.
HF_DATA_ROOT_PREFIX = "vector_store/"


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM Model Options ---
LLM_MODEL_OPTIONS = {
    "Flan-T5 Small (Default - CPU Friendly)": "google/flan-t5-small",
    "Flan-T5 Base (Larger, Slower on CPU)": "google/flan-t5-base",
    "Flan-T5 XL (Very Large, Likely OOM on Free Tier)": "google/flan-t5-xl",
    "Mistral-7B-Instruct-v0.2 (Large, Needs GPU)": "mistralai/Mistral-7B-Instruct-v0.2",
    "Falcon-7B-Instruct (Large, Needs GPU)": "tiiuae/falcon-7b-instruct",
    "Pythia-12B (Very Large, Needs GPU)": "OpenAssistant/oasst-sft-1-pythia-12b",
    "Zephyr-7B-Alpha (Large, Needs GPU)": "HuggingFaceH4/zephyr-7b-alpha",
    "Nous-Hermes-2-Mistral-7B-DPO (Large, Needs GPU)": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    "BART-Large-CNN (Summarization, May Fail)": "facebook/bart-large-cnn",
    "Mistral-7B-Instruct-GGUF (Special Format, Will Not Load!)": "TheBloke/Mistral-7B-Instruct-GGUF", 
}
DEFAULT_LLM_MODEL_KEY = "Flan-T5 Small (Default - CPU Friendly)"


TOP_K_RETRIEVAL = 5

# --- Quick Start Questions ---
QUICK_START_QUESTIONS = [
    "Select a question...", # Placeholder for initial selection
    "What are common issues with credit card billing?",
    "Tell me about problems with personal loan interest rates.",
    "What do customers complain about regarding Buy Now, Pay Later services?",
    "Are there issues accessing money from savings accounts?",
    "Describe typical problems with unauthorized money transfers.",
    "What kind of disputes arise from incorrect information on credit reports?",
    "How do customers complain about hidden fees in personal loans?",
    "What are the security concerns for money transfer services?",
    "Are there complaints about difficulty closing a credit card account?",
    "Summarize issues regarding delays in receiving funds from savings accounts."
]


# --- Function to Download Vector Store from Hugging Face Hub ---
@st.cache_resource
def download_vector_store_from_hf(repo_id: str, repo_type: str, local_base_dir: str, hf_data_root_prefix: str) -> bool:
    """
    Downloads vector store files from Hugging Face Hub to a local directory,
    preserving the subdirectory structure. Returns True on successful download and verification, False otherwise.
    """
    # Define the expected path for a key FAISS file after a successful download
    faiss_bin_target_path = os.path.join(local_base_dir, 'vector_store', 'faiss_index', 'faiss_index.bin')

    # Check if a key file already exists to determine if download is needed (for Streamlit caching)
    if os.path.exists(faiss_bin_target_path):
        print(f"Vector store already exists at {os.path.dirname(faiss_bin_target_path)}. Skipping download.")
        return True

    print(f"Attempting to download vector store from Hugging Face Hub '{repo_id}' under '{hf_data_root_prefix}' to '{local_base_dir}'...")
    
    try:
        # List all files in the repository matching the prefix
        all_files_in_repo = list_repo_files(repo_id=repo_id, repo_type=repo_type, revision="main")
        
        # Filter for files that start with the desired prefix and are not just the directory itself
        hf_file_paths_to_download = [
            f for f in all_files_in_repo
            if f.startswith(hf_data_root_prefix) and f != hf_data_root_prefix.rstrip('/')
        ]
        
        if not hf_file_paths_to_download:
            st.warning(f"No files found under prefix '{hf_data_root_prefix}' in '{repo_id}'. Please check your dataset content on Hugging Face Hub.")
            return False

    except Exception as e:
        st.error(f"Failed to list files from Hugging Face Hub '{repo_id}': {e}")
        st.error("Please ensure your HF_REPO_ID is correct and the dataset is accessible (check private repo if applicable).")
        return False

    download_count = 0
    download_success = True # Assume success unless a download fails
    for filename_in_repo in hf_file_paths_to_download:
        # The expected local path where this specific file will land
        expected_local_file_path = os.path.join(local_base_dir, filename_in_repo)
        
        # Ensure the local subdirectory exists before downloading the file
        os.makedirs(os.path.dirname(expected_local_file_path), exist_ok=True)

        print(f"Downloading '{filename_in_repo}' to expected local path '{expected_local_file_path}'...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename_in_repo,
                repo_type=repo_type,
                local_dir=local_base_dir,
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded '{filename_in_repo}' to '{downloaded_path}'")
            downloaded_files_count += 1
        except Exception as e:
            st.warning(f"Could not download '{filename_in_repo}': {e}")
            download_success = False # Mark overall download as failed if any file fails

    if downloaded_files_count > 0:
        print(f"Total files successfully downloaded: {downloaded_files_count}.")
    else:
        st.warning(f"No files were downloaded from Hugging Face Hub repo '{repo_id}' under prefix '{hf_data_root_prefix}'.")
        download_success = False # Mark as failure if no files were downloaded

    # Final verification after the download loop
    if not os.path.exists(faiss_bin_target_path):
        st.error(f"Final verification failed: Expected FAISS index file '{faiss_bin_target_path}' not found after download attempt.")
        download_success = False
    else:
        print(f"Final verification successful: FAISS index file '{faiss_bin_target_path}' exists.")

    return download_success


# --- Global RAG Pipeline Initialization ---
# This function will now also take the selected LLM model name
@st.cache_resource
def initialize_rag_pipeline(llm_model_name: str):
    # Attempt to download/verify the vector store files
    download_successful = download_vector_store_from_hf(HF_REPO_ID, HF_REPO_TYPE, LOCAL_VECTOR_STORE_DIR, HF_DATA_ROOT_PREFIX)

    if not download_successful:
        st.error("Vector store download or verification failed. Cannot initialize RAG Pipeline.")
        return None # Return None if download was not successful

    print(f"Initializing RAG components for Streamlit app with LLM: {llm_model_name}...")
    try:
        rag_retriever = Retriever(
            embedding_model_name=EMBEDDING_MODEL_NAME,
            faiss_index_path=FAISS_INDEX_PATH,
            faiss_metadata_path=FAISS_METADATA_PATH,
            chromadb_path=CHROMADB_PATH, # This is the local downloaded path
            chromadb_collection_name=CHROMADB_COLLECTION_NAME
        )
        # Pass the selected LLM model name to the Generator
        rag_generator = Generator(model_name=llm_model_name)
        st.success(f"Successfully loaded LLM: {llm_model_name}. Initializing RAG Pipeline...")
        rag_pipeline_instance = RAGPipeline(rag_retriever, rag_generator)
        print("RAG Pipeline initialized successfully for Streamlit app.")
        return rag_pipeline_instance
    except Exception as e:
        st.error(f"Failed to initialize RAG Pipeline with {llm_model_name}: {e}")
        st.error("This often happens with larger models due to insufficient memory (RAM) on Streamlit Community Cloud's free tier. Try a smaller model (e.g., 'Flan-T5 Small').")
        st.error("Please ensure vector stores are correctly downloaded and accessible to the Retriever. Check file paths and content integrity. Also, verify 'rag_pipeline.py' for any internal loading issues.")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="CrediTrust Complaint Chatbot", layout="centered")

st.title("CrediTrust Complaint Analysis Chatbot")
st.markdown("Ask questions about customer complaints related to financial products (Credit Card, Personal Loan, BNPL, Savings Account, Money Transfer).")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = ""
# Initialize selected LLM model in session state
if "selected_llm_model" not in st.session_state:
    st.session_state.selected_llm_model = DEFAULT_LLM_MODEL_KEY
# Initialize chat input value in session state
if "chat_input_value" not in st.session_state:
    st.session_state.chat_input_value = ""


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Quick Start Questions Dropdown ---
# Callback function to update the chat input when a quick question is selected
def update_chat_input_from_quick_question():
    selected_question = st.session_state.quick_question_selector
    if selected_question != QUICK_START_QUESTIONS[0]: # Avoid setting placeholder
        st.session_state.chat_input_value = selected_question
        # Clear the dropdown after selection to allow re-selection
        st.session_state.quick_question_selector = QUICK_START_QUESTIONS[0]


st.selectbox(
    "Quick Start Questions:",
    options=QUICK_START_QUESTIONS,
    index=0,
    key="quick_question_selector",
    on_change=update_chat_input_from_quick_question,
    help="Select a predefined question to populate the chat input."
)

# Input for new message
# The value is now controlled by st.session_state.chat_input_value
prompt = st.chat_input("Your question...", key="main_chat_input", value=st.session_state.chat_input_value)

if prompt:
    # Clear the chat input value in session state after it's used
    st.session_state.chat_input_value = "" # This prevents the input from sticking

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Re-initialize RAG pipeline with the currently selected LLM model
    rag_pipeline = initialize_rag_pipeline(LLM_MODEL_OPTIONS[st.session_state.selected_llm_model])
    RAG_INITIALIZED = (rag_pipeline is not None)

    # Generate AI response
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
                    st.session_session.sources = ""
                else:
                    result = rag_pipeline.run(prompt, k=TOP_K_RETRIEVAL, vector_store_type=vector_store_choice)
                    ai_response = result['answer']
                    st.session_state.sources = format_sources_for_display(result['sources'])
            
            st.markdown(ai_response)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    
    # LLM Model Selection Dropdown
    selected_model_display_name = st.selectbox(
        "Choose LLM Model:",
        options=list(LLM_MODEL_OPTIONS.keys()),
        index=list(LLM_MODEL_OPTIONS.keys()).index(st.session_state.selected_llm_model),
        key="llm_model_selector",
        help="Select the Large Language Model for generating answers."
    )
    # Update session state if a new model is selected
    if selected_model_display_name != st.session_state.selected_llm_model:
        st.session_state.selected_llm_model = selected_model_display_name
        # Clear chat and rerun when model changes to force re-initialization
        st.session_state.messages = []
        st.session_state.sources = ""
        st.rerun() # Rerun to apply new model selection

    st.warning(
        "**Important LLM Note:** Models like Mistral-7B, Falcon-7B, Pythia-12B, Flan-T5 XL, Zephyr-7B, and Nous-Hermes-2-Mistral-7B-DPO are very large (multi-GB) and **will likely cause out-of-memory errors or be extremely slow** on Streamlit Community Cloud's free CPU-only tier. "
        "For optimal performance with these powerful LLMs, **API-based solutions** (e.g., Gemini API, Hugging Face Inference Endpoints) on a paid cloud infrastructure are required. "
        "The **GGUF model ('TheBloke/Mistral-7B-Instruct-GGUF') will NOT load** with the current setup."
    )


    # Vector Store Selection
    options = ["faiss"]
    # Check if rag_pipeline is initialized before accessing its retriever
    if CHROMADB_AVAILABLE and rag_pipeline is not None and rag_pipeline.retriever.chroma_collection:
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
        st.rerun()

if st.session_state.sources:
    with st.expander("Show Retrieved Sources"):
        st.markdown(st.session_state.sources)