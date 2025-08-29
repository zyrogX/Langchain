# LangChain All Codes

This repository contains modular demos and experiments for building Retrieval-Augmented Generation (RAG) systems with LangChain. It includes document loaders, retrievers, text splitting strategies, vector stores, structured output parsing, and simple-to-parallel chain compositions.

## Project Structure

```
RAG/
  document_loader/           # PDF/Text/CSV/Web loaders
  retrievers/                # MMR, multi-query, Wikipedia, contextual compression
  text_spliter.py/           # length/recursive/semantic/code/markdown splitters
  vector_store/              # FAISS, Chroma examples and artifacts
chains/                      # simple, parallel, conditional chains
chatmodels/                  # OpenAI, Anthropic, Gemini, HF chat model wrappers
embedding_models/            # embeddings and similarity
llms/                        # LLM demos + audio transcription
output_parsars/              # JSON/Pydantic/structured output parsing
prompts/                     # prompt templates, history, chatbot demos
runnables/                   # sequences, branches, lambdas, passthrough
structured_output/           # schemas, typed dict, pydantic examples
requirements.txt
```

> Note: Some filenames/folders intentionally mirror the original learning setup and may contain typos (e.g., `text_spliter.py`). Keep them as-is for reproducibility.

## Features

* Document ingestion from PDFs, raw text, CSVs, and web pages
* Multiple retriever strategies: MMR, multi-query, Wikipedia, contextual compression
* Text chunking strategies: length-based, recursive, semantic, code-aware, markdown-aware
* Vector stores: FAISS and Chroma demos
* Runnable graph patterns: sequence, parallel, branch, lambda, passthrough
* Structured outputs with JSON Schema and Pydantic
* LLM integrations: OpenAI, Anthropic, Gemini, Hugging Face
* Basic audio transcription demo

## Getting Started

### 1) Clone

```bash
git clone https://github.com/zyrogX/Langchain.git
cd Langchain
```

### 2) Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Environment variables

Create a `.env` at the project root. Do not commit secrets.

```bash
# one-liner to start from a template once you add it
# cp .env.example .env   # macOS/Linux
# copy .env.example .env # Windows PowerShell
```

Example `.env` contents:

```
OPENAI_API_KEY=your_openai_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_gemini_key
```

### 5) Run examples

```bash
# Python scripts
python RAG/document_loader/csvloader.py
python chains/simple_chain.py

# Jupyter notebooks
pip install jupyter
jupyter notebook
# then open files under RAG/retrievers/, RAG/vector_store/, etc.
```

## Vector Stores and Large Artifacts

This repo may generate binary artifacts such as FAISS indexes (`*.faiss`, `*.pkl`) and Chroma segments (`*.bin`). These should generally not be committed. Prefer rebuilding them locally:

```python
# pseudo-example for FAISS
# from langchain_community.vectorstores import FAISS
# vs = FAISS.from_documents(docs, embeddings)
# vs.save_local("RAG/vector_store/faiss_ipl")
```


