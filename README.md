# **Intelligent Complaint Bot**

## **Project Description**

A Retrieval-Augmented Generation (RAG)-powered AI chatbot designed for CrediTrust Financial to streamline the analysis of customer complaints. This system transforms vast amounts of unstructured customer feedback into actionable insights, empowering product, support, and compliance teams to rapidly identify trends, address issues, and enhance service quality.

## **Table of Contents**

- [Project Description](#project-description)
- [Business Understanding](#business-understanding)
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Business Objectives](#business-objectives)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Live Demo](#live-demo)
- [Development and Evaluation](#development-and-evaluation)
- [Contributing](#contributing)
- [License](#license)
## **Business Understanding**

CrediTrust Financial, a rapidly expanding digital finance company in East Africa, faces the challenge of managing thousands of monthly customer complaints across various channels. Product Managers, like Asha, currently spend extensive hours manually sifting through these narratives to identify emerging issues. This manual, reactive process leads to delayed trend identification, overwhelmed support teams, and a lack of executive visibility into critical customer pain points. This project delivers an intelligent, automated solution to transform this raw data into a strategic asset.

## **Project Overview**

This project delivers an intelligent complaint-answering chatbot utilizing a robust Retrieval-Augmented Generation (RAG) architecture. The chatbot enables internal stakeholders to ask natural language questions about customer feedback and receive synthesized, evidence-backed answers. By combining advanced semantic search capabilities with a Large Language Model, the tool empowers teams to quickly understand customer pain points across various financial products, including credit cards, personal loans, BNPL, savings accounts, and money transfers. This shifts the company from reactive problem-solving to proactive identification and resolution of customer issues.

## **Key Features**

- **Natural Language Querying:** Users can ask plain-English questions about customer complaints (e.g., "Why are people unhappy with BNPL?").
- **Intelligent Retrieval (FAISS & ChromaDB):** The system employs a dual-vector database strategy, utilizing both FAISS and ChromaDB, alongside a `sentence-transformers/all-MiniLM-L6-v2` embedding model. This ensures efficient semantic search, retrieving the most relevant complaint narratives based on user queries.
- **LLM-Powered Answers:** The retrieved contextual information is fed into a Large Language Model (`google/flan-t5-small` for stable deployment) via robust prompt engineering. This generates concise, insightful, and grounded answers, minimizing hallucinations.
- **Multi-Product Analysis:** The system is capable of querying and analyzing issues across CrediTrust's five major financial product categories, providing a comprehensive view of customer sentiment.
- **Source Transparency:** To build user trust and enable verification, the original source text chunks used by the LLM to formulate its answer are explicitly displayed alongside the generated response.
- **Interactive Streamlit User Interface:** A user-friendly web interface built with Streamlit provides a seamless and intuitive interaction experience.
    - **Quick Start Questions:** A dropdown menu offers predefined common questions, allowing users to quickly explore the chatbot's capabilities and typical use cases.
    - **Clear Conversation:** A "Clear Chat" button allows users to reset the conversation history and sources display.
    - **Vector Store Selection:** Users can select between FAISS and ChromaDB for retrieval, enabling comparison of their performance (though ChromaDB may face limitations in certain deployment environments).

## **Business Objectives**

The success of the CrediTrust ComplaintBot will be measured against the following Key Performance Indicators (KPIs):

- **Decrease Time to Insight:** Reduce the time it takes for a Product Manager to identify a major complaint trend from days to minutes.
- **Empower Non-Technical Teams:** Enable non-technical teams (like Support and Compliance) to obtain answers to customer feedback questions without requiring a data analyst.
- **Shift to Proactive Problem Solving:** Facilitate a transition from reacting to problems to proactively identifying and fixing them based on real-time customer feedback.

## **Project Structure**

```
├── .vscode/                 # VSCode specific settings
├── .github/                 # GitHub specific configurations (e.g., Workflows)
│   └── workflows/
│       └── unittests.yml    # CI/CD workflow for tests and linting
├── .gitignore               # Specifies intentionally untracked files to ignore
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Modern Python packaging configuration (PEP 517/621)
├── README.md                # Project overview, installation, usage
├── Makefile                 # Common development tasks (setup, test, lint, clean)
├── .env                     # Environment variables (e.g., API keys - kept out of Git)
├── src/                     # Core source code for the project
│   ├── __init__.py          # Marks src as a Python package
│   ├── preprocessing.py     # Script for data loading, EDA, and cleaning
│   ├── vector_indexing.py   # Script for text chunking, embedding, and vector store indexing
│   ├── rag_pipeline.py      # Core RAG logic, including retriever and generator implementations
│   ├── streamlit_app.py     # The interactive chat interface application (Streamlit)
│   └── gradio_app.py        # The interactive chat interface application (Gradio - for reference)
│   └── static/              # Static assets for UI (e.g., avatars)
├── tests/                   # Test suite (unit, integration)
│   ├── unit/                # Unit tests for individual components
│   └── integration/         # Integration tests for combined components
├── notebooks/               # Jupyter notebooks for experimentation, EDA, prototyping
│   ├── data_processing_and_eda.ipynb           # Notebook for initial EDA and developing preprocessing steps
│   ├── chunking_embedding_experimentation.ipynb # Notebook for experimenting with chunking strategies and embedding models
│   └── rag_pipeline_and_evaluation.ipynb       # Notebook for qualitative evaluation and testing of the RAG pipeline
├── docs/                    # Project documentation (e.g., Sphinx docs)
├── data/                    # Data storage (raw, processed)
│   ├── raw/                 # Original, immutable raw data (e.g., raw CFPB dataset)
│   └── processed/           # Transformed, cleaned, or feature-engineered data (e.g., filtered_complaints.csv)
├── config/                  # Configuration files
└── vector_store/            # To persist your FAISS/ChromaDB vector store (uploaded to Hugging Face Hub)

```

## **Setup and Installation**

1. **Clone the repository:**
    
    ```
    git clone https://github.com/michaWorku/Intelligent-Compliant-Bot.git
    cd Intelligent-Compliant-Bot
    
    ```
    
2. **Create a virtual environment (recommended):**
    
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    
    ```
    
3. **Install dependencies:**
    
    ```
    pip install -r requirements.txt
    
    ```
    

## **Usage**

This project develops an intelligent complaint-answering chatbot for internal CrediTrust teams.

1. Prepare Data and Vector Store (Local Execution):
    
    To run the full pipeline locally and generate your vector store:
    
    ```
    python src/preprocessing.py
    python src/vector_indexing.py
    
    ```
    
    These scripts will process the data and build the vector store in the `vector_store/` directory.
    
2. **Run the Chatbot Interface (Local Execution):**
    
    ```
    streamlit run src/streamlit_app.py
    
    ```
    
    This will launch the interactive Streamlit web interface in your browser, allowing you to query the RAG system using locally built vector stores (if available).
    

## **Live Demo**

Experience the CrediTrust Complaint Analysis Chatbot live on Streamlit Community Cloud:

[**https://intelligent-compliant-chatbot.streamlit.app/**](https://intelligent-compliant-chatbot.streamlit.app/)

## **Development and Evaluation**

The project involved a structured development process, focusing on building a robust RAG pipeline and an intuitive user interface.

- **Data Processing and Preparation:** Initial exploration of the CFPB complaint dataset, handling missing values, filtering for relevant financial products, and comprehensive text cleaning were performed to ensure high-quality input.
- **Text Chunking and Vector Indexing:** Long complaint narratives were broken down into manageable chunks. These chunks were then converted into numerical vector embeddings using the `sentence-transformers/all-MiniLM-L6-v2` model. These embeddings were efficiently stored in both FAISS and ChromaDB vector databases for rapid semantic search. The large vector store files are hosted on Hugging Face Hub for cloud deployment.
- **RAG Core Logic Implementation:** The core Retrieval-Augmented Generation (RAG) pipeline was built. This involved developing a retriever to fetch relevant chunks based on user queries and integrating them with a Large Language Model (`google/flan-t5-small` for deployment stability) via carefully engineered prompts to generate insightful, context-aware answers.
- **Qualitative Evaluation:** A critical qualitative evaluation was conducted using a set of representative questions. This assessment analyzed the generated answers and retrieved sources, providing valuable insights into the system's effectiveness, identifying areas for improvement (e.g., retrieval precision for specific product categories, LLM synthesis quality), and guiding subsequent refinements.
- **Interactive Chat Interface Development:** A user-friendly web application was developed using Streamlit, enabling non-technical users to interact with the RAG system, ask questions, and view AI-generated answers along with their supporting sources. The interface includes quick-start questions and options for vector store selection.

## **Contributing**

Guidelines for contributing to the project.

## **License**

This project is licensed under the MIT License.