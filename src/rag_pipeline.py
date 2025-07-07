import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, set_seed
from typing import List, Dict, Union

# Try importing chromadb, but handle failure gracefully
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    print("Warning: chromadb not installed. ChromaDB functionality will be unavailable.")
    CHROMADB_AVAILABLE = False
except Exception as e:
    print(f"Warning: Error importing chromadb: {e}. ChromaDB functionality will be unavailable.")
    CHROMADB_AVAILABLE = False

# Set a seed for reproducibility if using models with randomness
set_seed(42)

class Retriever:
    """
    Handles loading of the embedding model and vector stores (FAISS and ChromaDB)
    and performs similarity search to retrieve relevant text chunks.
    """
    def __init__(self, embedding_model_name: str, faiss_index_path: str,
                 faiss_metadata_path: str, chromadb_path: str,
                 chromadb_collection_name: str = "complaint_chunks"):
        """
        Initializes the Retriever with an embedding model and loads vector stores.

        Args:
            embedding_model_name (str): Name of the Sentence Transformer model.
            faiss_index_path (str): Path to the saved FAISS index file.
            faiss_metadata_path (str): Path to the FAISS metadata CSV file.
            chromadb_path (str): Path to the ChromaDB persistent client directory.
            chromadb_collection_name (str): Name of the collection in ChromaDB.
        """
        print(f"Initializing Retriever...")
        self.embedding_model = self._load_embedding_model(embedding_model_name)
        
        self.faiss_index = None
        self.faiss_metadata = None
        self.chroma_collection = None

        # Load FAISS components
        if os.path.exists(faiss_index_path) and os.path.exists(faiss_metadata_path):
            self.faiss_index = self._load_faiss_index(faiss_index_path)
            self.faiss_metadata = self._load_faiss_metadata(faiss_metadata_path)
        else:
            print(f"Warning: FAISS index or metadata not found. FAISS retrieval will be unavailable. "
                  f"Checked: {faiss_index_path} and {faiss_metadata_path}")

        # Load ChromaDB components only if available and path exists
        if CHROMADB_AVAILABLE and os.path.exists(chromadb_path) and os.path.isdir(chromadb_path):
            try:
                self.chroma_client = chromadb.PersistentClient(path=chromadb_path)
                self.chroma_collection = self.chroma_client.get_collection(name=chromadb_collection_name)
                print(f"ChromaDB collection '{chromadb_collection_name}' loaded successfully. Count: {self.chroma_collection.count()}")
            except Exception as e:
                print(f"Warning: ChromaDB collection '{chromadb_collection_name}' not found or error loading: {e}. ChromaDB retrieval will be unavailable.")
                self.chroma_collection = None
        else:
            if CHROMADB_AVAILABLE: # Only print this if chromadb was imported but path was bad
                print(f"Warning: ChromaDB directory not found at {chromadb_path}. ChromaDB retrieval will be unavailable.")
            # Else, message already printed by CHROMADB_AVAILABLE check

        if self.embedding_model is None:
            raise ValueError("Embedding model failed to load. Retriever cannot function.")
        if self.faiss_index is None and self.chroma_collection is None:
            raise ValueError("Neither FAISS nor ChromaDB could be loaded. Retriever has no vector store.")

        print("Retriever initialized.")

    def _load_embedding_model(self, model_name: str):
        """Loads the sentence transformer embedding model."""
        print(f"Loading embedding model: {model_name}...")
        try:
            model = SentenceTransformer(model_name)
            print("Embedding model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading embedding model {model_name}: {e}")
            return None

    def _load_faiss_index(self, index_path: str):
        """Loads a FAISS index from disk."""
        print(f"Loading FAISS index from {index_path}...")
        try:
            index = faiss.read_index(index_path)
            print(f"FAISS index loaded with {index.ntotal} vectors.")
            return index
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return None

    def _load_faiss_metadata(self, metadata_path: str):
        """Loads FAISS metadata from a CSV file."""
        print(f"Loading FAISS metadata from {metadata_path}...")
        try:
            metadata_df = pd.read_csv(metadata_path)
            # Set 'chunk_id' as index for fast lookup if it's unique
            if 'chunk_id' in metadata_df.columns and not metadata_df['chunk_id'].duplicated().any():
                metadata_df.set_index('chunk_id', inplace=True)
            print(f"FAISS metadata loaded. Shape: {metadata_df.shape}")
            return metadata_df
        except Exception as e:
            print(f"Error loading FAISS metadata: {e}")
            return None

    def retrieve(self, query: str, k: int = 5, vector_store_type: str = "faiss") -> List[Dict]:
        """
        Performs a similarity search against the specified vector store.

        Args:
            query (str): The user's question.
            k (int): The number of top-k relevant chunks to retrieve.
            vector_store_type (str): 'faiss' or 'chromadb' to specify which store to use.

        Returns:
            List[Dict]: A list of dictionaries, each representing a retrieved chunk
                        with its text and metadata.
        """
        if self.embedding_model is None:
            print("Embedding model not loaded. Cannot retrieve.")
            return []

        query_embedding = self.embedding_model.encode([query]).astype('float32')

        retrieved_chunks = []

        if vector_store_type.lower() == "faiss":
            if self.faiss_index is None or self.faiss_metadata is None:
                print("FAISS index or metadata not loaded. Cannot perform FAISS retrieval.")
                return []
            
            print(f"Retrieving top {k} chunks from FAISS...")
            distances, indices = self.faiss_index.search(query_embedding, k)
            
            # FAISS returns indices, we need to map them back to our metadata
            for i, idx in enumerate(indices[0]):
                if idx < 0: # Handle cases where k is larger than available vectors
                    continue
                # Ensure we handle cases where metadata might be smaller than index.ntotal
                if idx >= len(self.faiss_metadata):
                    print(f"Warning: FAISS index {idx} out of bounds for metadata. Skipping.")
                    continue

                chunk_id_from_faiss = self.faiss_metadata.index[idx] if self.faiss_metadata.index.name == 'chunk_id' else self.faiss_metadata.iloc[idx]['chunk_id']
                
                # Retrieve the full chunk data using chunk_id
                chunk_data = self.faiss_metadata.loc[chunk_id_from_faiss].to_dict() if self.faiss_metadata.index.name == 'chunk_id' else \
                             self.faiss_metadata[self.faiss_metadata['chunk_id'] == chunk_id_from_faiss].iloc[0].to_dict()

                retrieved_chunks.append({
                    'text': chunk_data.get('text', ''),
                    'original_id': chunk_data.get('original_id', 'N/A'),
                    'product': chunk_data.get('product', 'N/A'),
                    'chunk_id': chunk_data.get('chunk_id', chunk_id_from_faiss),
                    'distance': float(distances[0][i]) # Add distance for analysis
                })
            print(f"FAISS retrieval complete. Found {len(retrieved_chunks)} chunks.")
        
        elif vector_store_type.lower() == "chromadb":
            if not CHROMADB_AVAILABLE or self.chroma_collection is None:
                print("ChromaDB not available or collection not loaded. Cannot perform ChromaDB retrieval.")
                return []
            
            print(f"Retrieving top {k} chunks from ChromaDB...")
            results = self.chroma_collection.query(
                query_embeddings=query_embedding.tolist(), # ChromaDB expects list of lists
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            if results and results['documents']:
                for i in range(len(results['documents'][0])):
                    retrieved_chunks.append({
                        'text': results['documents'][0][i],
                        'original_id': results['metadatas'][0][i].get('original_id', 'N/A'),
                        'product': results['metadatas'][0][i].get('product', 'N/A'),
                        'chunk_id': results['ids'][0][i],
                        'distance': results['distances'][0][i]
                    })
            print(f"ChromaDB retrieval complete. Found {len(retrieved_chunks)} chunks.")

        else:
            print(f"Error: Unknown vector store type '{vector_store_type}'. Please choose 'faiss' or 'chromadb'.")
            return []
            
        return retrieved_chunks

class Generator:
    """
    Handles the generation of responses using a Large Language Model (LLM)
    based on a prompt, user question, and retrieved context.
    """
    def __init__(self, model_name: str = "google/flan-t5-small"):
        """
        Initializes the Generator with an LLM.

        Args:
            model_name (str): Name of the Hugging Face model for text generation.
                              e.g., "google/flan-t5-small", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        """
        print(f"Initializing Generator with model: {model_name}...")
        try:
            import torch # Import torch here to ensure it's available for pipeline
            self.llm_pipeline = pipeline("text2text-generation", model=model_name, device=0 if torch.cuda.is_available() else -1)
            print("LLM pipeline loaded successfully.")
        except ImportError:
            print("Warning: PyTorch not installed. LLM pipeline will use a dummy response.")
            self.llm_pipeline = None
        except Exception as e:
            print(f"Error loading LLM model {model_name}: {e}")
            self.llm_pipeline = None
            print("Warning: LLM pipeline failed to load. Generator will use a dummy response.")

    def generate_response(self, question: str, context: str) -> str:
        """
        Generates an answer using the LLM based on the provided question and context.

        Args:
            question (str): The user's question.
            context (str): The retrieved relevant text chunks.

        Returns:
            str: The LLM's generated answer.
        """
        if self.llm_pipeline is None:
            return "I am sorry, my language model is not available at the moment. Please try again later."

        # Prompt Template
        prompt_template = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
        Use the following retrieved complaint excerpts to formulate your answer.
        If the context doesn't contain the answer, state that you don't have enough information.

        Context: {context}
        Question: {question}
        Answer:"""

        full_prompt = prompt_template.format(context=context, question=question)

        print("Generating response with LLM...")
        try:
            response = self.llm_pipeline(full_prompt, max_new_tokens=200, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)[0]['generated_text']
            print("LLM response generated.")
            return response.strip()
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return "I encountered an error while generating a response."

class RAGPipeline:
    """
    Orchestrates the retrieval and generation process.
    """
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator
        if retriever is None or generator is None:
            raise ValueError("Retriever and Generator must be initialized before creating RAGPipeline.")
        print("RAGPipeline initialized.")

    def run(self, query: str, k: int = 5, vector_store_type: str = "faiss") -> Dict[str, Union[str, List[Dict]]]:
        """
        Executes the full RAG pipeline.

        Args:
            query (str): The user's question.
            k (int): Number of chunks to retrieve.
            vector_store_type (str): 'faiss' or 'chromadb'.

        Returns:
            Dict[str, Union[str, List[Dict]]]: A dictionary containing the generated answer
                                              and the list of retrieved sources.
        """
        print(f"\n--- Running RAG Pipeline for query: '{query}' ---")
        
        # 1. Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(query, k=k, vector_store_type=vector_store_type)
        
        if not retrieved_chunks:
            print("No relevant chunks retrieved. Cannot generate an answer.")
            return {
                "answer": "I don't have enough information to answer your question based on the available complaints.",
                "sources": []
            }

        # Concatenate retrieved texts to form the context for the LLM
        context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
        
        # 2. Generate response
        answer = self.generator.generate_response(query, context)
        
        print("--- RAG Pipeline Complete ---")
        return {
            "answer": answer,
            "sources": retrieved_chunks
        }

if __name__ == "__main__":
    # --- Configuration for standalone script execution (adjust paths for your setup) ---
    # In Google Colab, these paths will be relative to your mounted Google Drive
    # e.g., if your project root is '/content/drive/My Drive/Colab_Project/'
    
    # Define paths relative to the project root (assuming you run from PROJECT_ROOT)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, 'vector_store', 'faiss_index', 'faiss_index.bin')
    FAISS_METADATA_PATH = os.path.join(PROJECT_ROOT, 'vector_store', 'faiss_index', 'faiss_metadata.csv')
    CHROMADB_PATH = os.path.join(PROJECT_ROOT, 'vector_store', 'chroma_db')

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "google/flan-t5-small" # Or a larger model if you have GPU resources

    print("Attempting to initialize RAG components for testing...")
    try:
        retriever = Retriever(EMBEDDING_MODEL, FAISS_INDEX_PATH, FAISS_METADATA_PATH, CHROMADB_PATH)
        generator = Generator(LLM_MODEL)
        rag_pipeline = RAGPipeline(retriever, generator)

        # Example usage
        test_query = "What are common complaints about credit card billing?"
        
        print("\n--- Testing with FAISS ---")
        result_faiss = rag_pipeline.run(test_query, k=3, vector_store_type="faiss")
        print("\nGenerated Answer (FAISS):")
        print(result_faiss['answer'])
        print("\nTop Source (FAISS):")
        if result_faiss['sources']:
            print(f"Product: {result_faiss['sources'][0]['product']}, ID: {result_faiss['sources'][0]['original_id']}")
            print(result_faiss['sources'][0]['text'][:200] + "...") # Print first 200 chars

        print("\n--- Testing with ChromaDB ---")
        if CHROMADB_AVAILABLE and retriever.chroma_collection: # Only test if ChromaDB loaded successfully
            result_chroma = rag_pipeline.run(test_query, k=3, vector_store_type="chromadb")
            print("\nGenerated Answer (ChromaDB):")
            print(result_chroma['answer'])
            print("\nTop Source (ChromaDB):")
            if result_chroma['sources']:
                print(f"Product: {result_chroma['sources'][0]['product']}, ID: {result_chroma['sources'][0]['original_id']}")
                print(result_chroma['sources'][0]['text'][:200] + "...") # Print first 200 chars
        else:
            print("ChromaDB not available or not initialized, skipping test.")


    except Exception as e:
        print(f"\nError during RAG pipeline test: {e}")
        print("Please ensure Task 1 and Task 2 were completed successfully and all files are in place.")