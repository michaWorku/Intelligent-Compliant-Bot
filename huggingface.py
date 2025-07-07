from huggingface_hub import HfApi
import os

# Replace with your Hugging Face username and the dataset name you created
HF_USERNAME = "michaWorku" # Your username
HF_DATASET_NAME = "credittrust-rag" # Your dataset name

# Path to your local vector_store directory
LOCAL_VECTOR_STORE_PATH = "data" # This is the folder you want to upload

# Initialize the HfApi
api = HfApi()

print(f"Uploading {LOCAL_VECTOR_STORE_PATH} to {HF_USERNAME}/{HF_DATASET_NAME}/data/...")

# Upload the entire folder. The folder_path will be placed under the repo_id.
# The 'path_in_repo' argument ensures it's placed inside 'vector_store/'
api.upload_folder(
    folder_path=LOCAL_VECTOR_STORE_PATH,
    repo_id=f"{HF_USERNAME}/{HF_DATASET_NAME}",
    repo_type="dataset",
    path_in_repo="data", # This tells HF to put the contents of LOCAL_VECTOR_STORE_PATH into 'vector_store/'
    commit_message="Uploading customer compliants raw and processed data"
)

print("Upload complete!")