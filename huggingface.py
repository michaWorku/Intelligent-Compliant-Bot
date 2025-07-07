from huggingface_hub import HfApi
import os

# Replace with your Hugging Face username and the dataset name you created
HF_USERNAME = "michaWorku"
HF_DATASET_NAME = "credittrust-rag"

# Path to your local vector_store directory
LOCAL_VECTOR_STORE_PATH = "vector_store" # This is relative to where you run this script

# Initialize the HfApi
api = HfApi()

print(f"Uploading {LOCAL_VECTOR_STORE_PATH} to {HF_USERNAME}/{HF_DATASET_NAME}...")

# Upload the entire folder
api.upload_folder(
    folder_path=LOCAL_VECTOR_STORE_PATH,
    repo_id=f"{HF_USERNAME}/{HF_DATASET_NAME}",
    repo_type="dataset",
    commit_message="Initial upload of RAG vector store files"
)

print("Upload complete!")