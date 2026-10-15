# RAG-Mini-Wikipedia Evaluation

This project implements a retrieval-augmented generation (RAG) question-answering pipeline on the `rag-mini-wikipedia` dataset. It compares a **Base model** (answering from internal knowledge), a **RAG model** (retrieving and adding evidence to context), and a **RAG + Reranker model** (retrieving and reranking evidence before generation).

## Key Features
- **RAG Pipeline**: ChromaDB-based dense retrieval using `all-MiniLM-L6-v2`.
- **Reranker**: Optionally uses a `cross-encoder/ms-marco-MiniLM-L6-v2` to re-rank retrieved documents.
- **Comparative Evaluation**: Side-by-side analysis of "Base" vs. "RAG" vs. "RAG-Reranked" performance.
- **LLM-as-a-Judge**: Automated scoring of generated text using a dedicated judge model (Gemma-4-E4B).

## System Architecture
The pipeline follows a standard RAG workflow optimized for evaluation:
1. **Ingestion**: Wikipedia passages are embedded using `all-MiniLM-L6-v2` and stored in **ChromaDB**.
2. **Querying**: User questions are converted into vectors, and the system retrieves the top-5 most similar passages.
3. **Reranking (Optional)**: When enabled, the retriever's results are re-scored using a Cross-Encoder to select the most relevant documents.
4. **Augmentation**: Retrieved and/or reranked passages are injected into the prompt for the generator.
5. **Generation**: **Qwen/Qwen3-4B** generates an answer based on the provided context.
6. **Judging**: **google/gemma-4-E4B-it** compares the output against the question and ground truth and responds with Yes/No depending on their similarity.

## Evaluation & Results
- **Methodology**: A separate judge model evaluates every prediction from "Base", "RAG", and "RAG-Reranked" modes.
- **Outputs**: The project generates `res-base.csv`, `res-rag.csv`, and `res-rag-reranked.csv`, along with their corresponding judge-scored files.
- **Key Findings**: 
  - **Base Mode (No Retrieval)**: 435/918 correct (~47.39%)
  - **RAG Mode (With Retrieval)**: 736/918 correct (~80.17%)
  - **RAG-Reranked Mode**: 734/918 correct (~79.96%)

## Installation & Usage
### Installation
```bash
uv sync
```

### Running Evaluation
Run the inference for any mode:
```bash
uv run python main.py --mode [base|rag|rag-reranked]
```
Run the automated judge:
```bash
uv run python llm-judge.py
```

Run the `result_analysis.ipynb` notebook to compare results (correct answers).

## Tech Stack
- **Models**: Qwen/Qwen3-4B (Generator), google/gemma-4-E4B-it (Judge), cross-encoder/ms-marco-MiniLM-L6-v2 (Reranker)
- **Vector Database**: ChromaDB
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Frameworks**: PyTorch, Hugging Face Transformers, Pandas
