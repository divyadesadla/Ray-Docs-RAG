# Ray Documentation RAG Pipeline

This project provides a comprehensive pipeline for scraping, extracting, chunking, embedding, and storing the Ray documentation in Pinecone. The goal is to facilitate efficient and accurate querying of Ray documentation using a **Retrieval-Augmented Generation (RAG)** framework. The pipeline leverages modern tools like **Ray**, **Pinecone**, and **MiniLM** to provide a scalable and performant solution for working with large datasets.

## Overview

The Ray Documentation RAG Pipeline automates the process of downloading, processing, chunking, embedding, and storing the Ray documentation to enable efficient retrieval of information. This pipeline is designed to work with large datasets and improve querying by breaking down content into smaller, manageable pieces, embedding them in vector space, and storing them in Pinecone for fast retrieval.

Key Steps in the Pipeline:
1. **Scraping**: Download the entire Ray documentation website for offline access.
2. **Extraction**: Parse the HTML files and extract meaningful sections of text.
3. **Chunking**: Break down extracted content into smaller chunks for efficient processing.
4. **Embedding**: Use a pre-trained model (MiniLM) to embed the text chunks.
5. **Storing**: Store the embedded text chunks in Pinecone, a vector database, for fast retrieval during query processing.

The pipeline also includes a **RAG model** for querying the documentation and generating relevant responses.

## Features

- **Scalable Scraping**: Download the full Ray documentation website with a recursive scrape.
- **Efficient Extraction**: Use BeautifulSoup to extract meaningful content from HTML pages.
- **Text Chunking**: Break large sections of content into smaller, manageable chunks to improve processing.
- **Vector Embedding**: Embed text chunks using the `MiniLM` model to represent content in vector space.
- **Fast Retrieval**: Store embeddings in Pinecone for quick similarity-based searches.
- **Parallel Processing**: Use **Ray** for parallel processing to scale operations and handle large datasets efficiently.

## Pipeline Components

### 1. Scraping the Ray Documentation (`scraping.py`)

This script downloads a full offline copy of the Ray documentation website using the `wget` command, preserving the structure and resources of the site.

#### Features:
- Recursively downloads files from the Ray documentation site.
- Converts links to be relative for offline use.
- Handles HTML, images, stylesheets, and other resources.
- Allows customization of the output directory.

### 2. Extracting Content (`extraction.py`)

The extraction script processes the downloaded HTML files and extracts meaningful sections of text using **BeautifulSoup**. The extracted content is structured for further processing in the RAG pipeline.

#### Features:
- Recursively scans and extracts content from HTML files.
- Captures clean paragraph text and section identifiers.
- Retains the full URL relative to the original Ray documentation site.

### 3. Chunking the Extracted Content (`chunking.py`)

The chunking script splits large content sections into smaller chunks using **langchain's** `RecursiveCharacterTextSplitter`. This enables more efficient embedding and querying of the documentation.

#### Features:
- Splits long text sections into smaller, manageable chunks.
- Uses Ray for parallel processing of chunking tasks.
- Retains metadata (such as section IDs and URLs) with each chunk.
- Configurable chunk sizes and overlaps to fit specific needs.

### 4. Embedding and Storing in Pinecone (`miniLM_vector_store.py`)

This script embeds text chunks using the `MiniLM` model and stores the embeddings in **Pinecone**, a vector database, for fast retrieval.

#### Features:
- Initializes Pinecone and creates an index to store the embeddings.
- Uses **SentenceTransformers** to generate embeddings.
- Handles metadata truncation to fit within Pinecone's size limits.
- Batch processing for efficient embedding and storage.

### 5. Full Pipeline Execution (`pipeline.py`)

This is the master script that runs the full pipeline, including scraping, extraction, chunking, embedding, and storing embeddings in Pinecone.

#### Key Functions:
- **`safe_process(item)`**: Handles safe processing of each HTML file.
- **`run_pipeline()`**: Orchestrates the entire pipeline, processing HTML files, extracting content, chunking, embedding, and storing in Pinecone.

### 6. Response Generation with RAG (`rag_chain.py`)

This script integrates the RAG model with the pipeline. It retrieves context from Pinecone, feeds it into a language model, and generates responses based on the extracted documentation.

#### Key Functions:
- **`safe_process(item)`**: Ensures safe processing for each HTML file and section.
- **`run_pipeline()`**: Orchestrates the process of extracting content, chunking, and indexing in Pinecone.

### 7. Response Evaluation (`rag_eval.py`)

This script evaluates the performance of the generated responses, comparing them with ground-truth data using various metrics such as **BLEU**, **ROUGE**, and **embedding similarity**.

#### Features:
- **`evaluate_response()`**: Evaluates response relevance, faithfulness, completeness, fluency, and similarity.
- **`evaluate_rag_vs_llm()`**: Compares RAG and LLM-generated responses.
- **`save_eval_report()`**: Saves evaluation results for further analysis.

#### Evaluation Metrics:
- **Relevance**: Measures how relevant the response is to the query.
- **Faithfulness**: Assesses how faithful the response is to the context.
- **Completeness**: Evaluates whether the response covers the necessary content.
- **Fluency**: Measures the response's grammatical quality.
- **BLEU (Bilingual Evaluation Understudy)**: Evaluates n-gram overlap between the generated response and context.
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: Measures recall-based overlap between the response and reference context.
    ROUGE-1: Overlap of unigrams (single words).
    ROUGE-2: Overlap of bigrams (two words together).
    ROUGE-L: Measures the longest common subsequence overlap.
- **Embedding Similarity**: Measures cosine similarity between the context and the generated response.

## Setup

### Requirements

- Python 3.7+
- Required Python libraries:
  - `requests`, `beautifulsoup4`, `langchain`, `ray`, `pinecone-client`, `sentence-transformers`, `dotenv`
- Pinecone API key (for vector storage) and OPENAI_API_KEY (if running openai_vector_store.py instead of miniLM_vector_store.py) and save in .env file in src/
- Hugging Face model for embeddings (`MiniLM` or other models)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ray-docs-rag-pipeline.git
   cd ray-docs-rag-pipeline

2. Run the code in order of:
   a. scraping.py
   b. pipieline.py
   c. rag_chain.py
   d. rag_eval.py