# Intelligent Complaint Bot

## Table of Contents
- [Project Description](#project-description)
- [Business Understanding](#business-understanding)
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Business Objectives](#business-objectives)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Development and Testing](#development-and-testing)
- [Contributing](#contributing)
- [Team](#team)
- [License](#license)

## Project Description
A RAG-powered AI chatbot for CrediTrust Financial to analyze customer complaints. Transforms unstructured feedback into actionable insights, helping product, support, and compliance teams quickly identify trends and issues.

## Business Understanding

CrediTrust Financial, a rapidly growing digital finance company in East Africa, faces a significant challenge: managing thousands of customer complaints received monthly across various channels. Product Managers like Asha currently spend countless hours manually sifting through these unstructured complaint narratives to identify emerging issues. This manual process leads to delayed trend identification, overwhelmed support teams, reactive compliance, and a lack of executive visibility into critical customer pain points. The need for an intelligent, automated solution to transform this raw data into a strategic asset is paramount.

## Project Overview

This project aims to develop an intelligent complaint-answering chatbot utilizing Retrieval-Augmented Generation (RAG) to address CrediTrust Financial's challenges. The chatbot will enable internal stakeholders (Product, Support, Compliance) to ask natural language questions about customer feedback and receive synthesized, evidence-backed answers. By combining semantic search with large language models, the tool will empower teams to quickly understand customer pain points across credit cards, personal loans, BNPL, savings accounts, and money transfers, shifting the company from reactive problem-solving to proactive identification and resolution.

## Key Features

* **Natural Language Querying:** Allows users to ask plain-English questions about customer complaints (e.g., "Why are people unhappy with BNPL?").
* **Semantic Search:** Utilizes a vector database (FAISS or ChromaDB) and embedding models to retrieve the most semantically relevant complaint narratives.
* **LLM-Powered Answers:** Feeds retrieved context into a Large Language Model (LLM) to generate concise, insightful, and grounded answers.
* **Multi-Product Analysis:** Supports querying and comparison of issues across CrediTrust's five major financial product categories.
* **Source Transparency:** Displays the original source text chunks used by the LLM to build user trust and allow for verification.
* **Interactive User Interface:** A user-friendly web interface built with Gradio or Streamlit for seamless interaction.

## Business Objectives

The success of the CrediTrust ComplaintBot will be measured against the following Key Performance Indicators (KPIs):

* **Decrease Time to Insight:** Reduce the time it takes for a Product Manager to identify a major complaint trend from days to minutes.
* **Empower Non-Technical Teams:** Enable non-technical teams (like Support and Compliance) to obtain answers to customer feedback questions without requiring a data analyst.
* **Shift to Proactive Problem Solving:** Facilitate a transition from reacting to problems to proactively identifying and fixing them based on real-time customer feedback.

## Project Structure

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
│   ├── preprocessing.py     # Script for data loading, EDA, and cleaning (Task 1)
│   ├── vector_indexing.py   # Script for text chunking, embedding, and vector store indexing (Task 2)
│   ├── rag_pipeline.py      # Core RAG logic, including retriever and generator implementations (Task 3)
│   └── app.py               # The interactive chat interface application (Task 4)
├── tests/                   # Test suite (unit, integration)
│   ├── unit/                # Unit tests for individual components
│   └── integration/         # Integration tests for combined components
├── notebooks/               # Jupyter notebooks for experimentation, EDA, prototyping
│   ├── eda_and_preprocessing.ipynb           # Notebook for initial EDA and developing preprocessing steps (Task 1)
│   ├── chunking_embedding_experimentation.ipynb # Notebook for experimenting with chunking strategies and embedding models (Task 2)
│   └── rag_evaluation.ipynb                  # Notebook for qualitative evaluation and testing of the RAG pipeline (Task 3)
├── docs/                    # Project documentation (e.g., Sphinx docs)
├── data/                    # Data storage (raw, processed)
│   ├── raw/                 # Original, immutable raw data (e.g., raw CFPB dataset)
│   └── processed/           # Transformed, cleaned, or feature-engineered data (e.g., filtered_complaints.csv)
├── config/                  # Configuration files
└── vector_store/            # To persist your FAISS/ChromaDB vector store
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/michaWorku/Intelligent-Compliant-Bot.git](https://github.com/michaWorku/Intelligent-Compliant-Bot.git) # Update this URL
    cd Intelligent-Compliant-Bot
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

This project develops an intelligent complaint-answering chatbot for internal CrediTrust teams.

1.  **Run Data Preprocessing:**
    ```bash
    python src/preprocessing.py
    ```
    This script will load, clean, and save the filtered complaint data to `data/processed/filtered_complaints.csv`.

2.  **Build Vector Store:**
    ```bash
    python src/vector_indexing.py
    ```
    This script will chunk the cleaned text narratives, generate embeddings, and store them in the vector database in `vector_store/`.

3.  **Run the Chatbot Interface:**
    ```bash
    python src/app.py
    ```
    This will launch the interactive web interface (Gradio/Streamlit) in your browser, allowing you to query the RAG system.

## Development and Testing

The project development follows a structured approach across four key tasks:

* **Task 1: Exploratory Data Analysis and Data Preprocessing (`src/preprocessing.py`, `notebooks/eda_and_preprocessing.ipynb`)**: Focuses on understanding the raw CFPB complaint data, handling missing values, filtering for relevant products, and cleaning text narratives to ensure high-quality input for the RAG pipeline.
* **Task 2: Text Chunking, Embedding, and Vector Store Indexing (`src/vector_indexing.py`, `notebooks/chunking_embedding_experimentation.ipynb`)**: Involves breaking down long complaint narratives into manageable chunks, generating numerical vector embeddings using a chosen model (e.g., `sentence-transformers/all-MiniLM-L6-v2`), and storing these embeddings in a vector database (FAISS or ChromaDB) for efficient semantic search.
* **Task 3: Building the RAG Core Logic and Evaluation (`src/rag_pipeline.py`, `notebooks/rag_evaluation.ipynb`)**: Implements the retrieval mechanism to fetch relevant chunks based on user queries and integrates them with a Large Language Model (LLM) via robust prompt engineering to generate insightful answers. This task also includes a critical qualitative evaluation phase to assess the system's effectiveness.
* **Task 4: Creating an Interactive Chat Interface (`src/app.py`)**: Develops a user-friendly web application using Gradio or Streamlit, enabling non-technical users to interact with the RAG system, ask questions, and view AI-generated answers along with their supporting sources.

Testing will primarily involve qualitative evaluation (Task 3) to ensure the RAG pipeline provides accurate, relevant, and grounded responses. Unit tests for individual modules (`src/`) can be added in the `tests/` directory as development progresses.

## Contributing

Guidelines for contributing to the project.

## License

This project is licensed under the MIT License.
