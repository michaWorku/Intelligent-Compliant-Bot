import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from tqdm.notebook import tqdm # For progress bars in Colab notebooks

tqdm.pandas() # Enable pandas progress_apply

def load_cleaned_data(file_path: str) -> pd.DataFrame:
    """Loads the cleaned complaint data from a CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: Cleaned data file not found at {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        print(f"Cleaned data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading cleaned data: {e}")
        return None

def chunk_text(df: pd.DataFrame, narrative_column: str, id_column: str, product_column: str,
               chunk_size: int = 500, chunk_overlap: int = 100):
    """
    Chunks the text narratives into smaller segments, retaining original metadata.

    Args:
        df (pd.DataFrame): DataFrame containing the cleaned narratives.
        narrative_column (str): Name of the column containing the cleaned narratives.
        id_column (str): Name of the column containing the complaint ID.
        product_column (str): Name of the column containing the product category.
        chunk_size (int): The maximum size of each text chunk (in characters).
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        list: A list of dictionaries, where each dictionary represents a chunk
              and includes 'text', 'original_id', 'product', and 'chunk_id'.
    """
    if df is None or df.empty or narrative_column not in df.columns or \
       id_column not in df.columns or product_column not in df.columns:
        print("Invalid DataFrame or missing required columns for chunking.")
        return []

    print(f"Starting text chunking with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True, # Helps in tracing if needed
    )

    all_chunks = []
    chunk_counter = 0

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing narratives for chunking"):
        narrative = str(row[narrative_column]).strip()
        if not narrative:
            continue

        original_id = row[id_column]
        product = row[product_column]

        # LangChain's create_documents expects a list of texts and metadata
        # We need to map our DataFrame rows to LangChain Document objects
        # For simplicity here, we'll directly generate chunks with custom metadata
        # If using LangChain's Document class, you'd do:
        # docs = [Document(page_content=narrative, metadata={"original_id": original_id, "product": product})]
        # chunks = text_splitter.split_documents(docs)

        # Manually splitting and adding metadata for fine-grained control
        chunks_for_narrative = text_splitter.split_text(narrative)
        for i, chunk_text_content in enumerate(chunks_for_narrative):
            all_chunks.append({
                'text': chunk_text_content,
                'original_id': original_id,
                'product': product,
                'chunk_id': f"{original_id}_chunk_{i}", # Unique ID for each chunk
            })
            chunk_counter += 1
    print(f"Finished chunking. Total chunks created: {len(all_chunks)}")
    return all_chunks

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Loads the sentence transformer embedding model."""
    print(f"Loading embedding model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print("Embedding model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading embedding model {model_name}: {e}")
        return None

def create_embeddings(chunks: list, model):
    """Generates embeddings for a list of text chunks."""
    if not chunks:
        print("No chunks to embed.")
        return np.array([])
    if model is None:
        print("Embedding model not loaded. Cannot create embeddings.")
        return np.array([])

    print(f"Generating embeddings for {len(chunks)} chunks...")
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    print("Embeddings generated.")
    return embeddings

def create_and_save_faiss_index(embeddings: np.ndarray, chunks: list, save_path: str):
    """
    Creates a FAISS index from embeddings and saves it along with metadata.
    Metadata is saved as a separate CSV file because FAISS itself doesn't store it.
    """
    if embeddings.size == 0:
        print("No embeddings to index for FAISS.")
        return

    print(f"Creating FAISS index with {embeddings.shape[0]} embeddings, dimension {embeddings.shape[1]}...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # Using L2 distance for similarity search
    index.add(embeddings)
    print("FAISS index created.")

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save FAISS index
    faiss_index_file = os.path.join(save_path, "faiss_index.bin")
    faiss.write_index(index, faiss_index_file)
    print(f"FAISS index saved to {faiss_index_file}")

    # Save metadata
    metadata_df = pd.DataFrame([
        {'chunk_id': chunk['chunk_id'], 'original_id': chunk['original_id'],
         'product': chunk['product'], 'text': chunk['text']}
        for chunk in chunks
    ])
    metadata_file = os.path.join(save_path, "faiss_metadata.csv")
    metadata_df.to_csv(metadata_file, index=False)
    print(f"FAISS metadata saved to {metadata_file}")
    print("FAISS indexing complete.")

def create_and_save_chromadb_collection(embeddings: np.ndarray, chunks: list, db_path: str, collection_name: str = "complaint_chunks"):
    """
    Creates a ChromaDB collection and stores embeddings with metadata.
    """
    if embeddings.size == 0:
        print("No embeddings to index for ChromaDB.")
        return

    print(f"Creating ChromaDB collection '{collection_name}' at {db_path}...")
    client = chromadb.PersistentClient(path=db_path)
    # Ensure embedding function is set if using a custom model, or use default for pre-computed embeddings
    # For pre-computed embeddings, ChromaDB expects ids, embeddings, metadatas, and documents
    
    # Remove existing collection if it exists to ensure a clean run
    try:
        client.delete_collection(name=collection_name)
        print(f"Existing collection '{collection_name}' deleted.")
    except Exception as e:
        print(f"No existing collection '{collection_name}' to delete, or error during deletion: {e}")

    collection = client.create_collection(name=collection_name)
    
    ids = [chunk['chunk_id'] for chunk in chunks]
    documents = [chunk['text'] for chunk in chunks]
    metadatas = [{'original_id': chunk['original_id'], 'product': chunk['product']} for chunk in chunks]

    # Batching for large datasets
    batch_size = 5000  # Adjust based on your RAM
    for i in tqdm(range(0, len(ids), batch_size), desc="Adding chunks to ChromaDB"):
        batch_ids = ids[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size].tolist() # Convert numpy array to list of lists
        batch_metadatas = metadatas[i:i + batch_size]
        batch_documents = documents[i:i + batch_size]

        collection.add(
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_documents,
            ids=batch_ids
        )
    print(f"ChromaDB collection '{collection_name}' created and populated.")
    print("ChromaDB indexing complete.")

def main_indexing_process(cleaned_data_path: str, faiss_save_dir: str, chromadb_save_dir: str,
                          narrative_col: str, id_col: str, product_col: str,
                          chunk_size: int, chunk_overlap: int, embedding_model_name: str):
    """Main function to orchestrate the chunking, embedding, and indexing process."""
    print("--- Starting Chunking, Embedding, and Indexing Process ---")

    # 1. Load Cleaned Data
    df_cleaned = load_cleaned_data(cleaned_data_path)
    if df_cleaned is None:
        return

    # 2. Chunk Text Narratives
    # Ensure the cleaned narrative column is string type and handle potential NaNs before chunking
    df_cleaned[narrative_col] = df_cleaned[narrative_col].astype(str).fillna('')
    chunks = chunk_text(df_cleaned, narrative_col, id_col, product_col, chunk_size, chunk_overlap)
    if not chunks:
        print("No chunks generated. Exiting.")
        return

    # 3. Load Embedding Model
    model = get_embedding_model(embedding_model_name)
    if model is None:
        return

    # 4. Generate Embeddings
    embeddings = create_embeddings(chunks, model)
    if embeddings.size == 0:
        print("No embeddings generated. Exiting.")
        return

    # 5. Create and Save FAISS Index
    create_and_save_faiss_index(embeddings, chunks, faiss_save_dir)

    # 6. Create and Save ChromaDB Collection
    create_and_save_chromadb_collection(embeddings, chunks, chromadb_save_dir)

    print("--- Chunking, Embedding, and Indexing Process Complete ---")

if __name__ == "__main__":
    # --- Configuration for standalone script execution (adjust paths for your setup) ---
    # In Google Colab, these paths will be relative to your mounted Google Drive
    # e.g., if your project root is '/content/drive/My Drive/Colab_Project/'
    # and cleaned_complaints.csv is in 'Colab_Project/data/processed/'
    
    # Example paths (will be passed from notebook for Colab)
    CLEANED_DATA_PATH = 'data/processed/filtered_complaints.csv'
    FAISS_SAVE_DIR = 'vector_store/faiss_index'
    CHROMADB_SAVE_DIR = 'vector_store/chroma_db'

    NARRATIVE_COLUMN = 'cleaned_Consumer complaint narrative' # Name of the cleaned narrative column from Task 1
    ID_COLUMN = 'Complaint ID' # Or whatever your ID column is named
    PRODUCT_COLUMN = 'Product' # Or whatever your Product column is named

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    # For running this script directly (not in Colab via notebook)
    # Ensure you are in the project root or adjust paths accordingly.
    # main_indexing_process(CLEANED_DATA_PATH, FAISS_SAVE_DIR, CHROMADB_SAVE_DIR,
    #                       NARRATIVE_COLUMN, ID_COLUMN, PRODUCT_COLUMN,
    #                       CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME)
    print("This script is primarily designed to be called from the Colab notebook.")
    print("Please run the 'notebooks/chunking_embedding_and_indexing.ipynb' notebook.")