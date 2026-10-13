# RAG-Mini-Wikipedia Evaluation

This project implements a retrieval-augmented generation (RAG) question-answering pipeline on the `rag-mini-wikipedia` dataset. It compares a **Base model** (answering from internal knowledge) against a **RAG model** (retrieving and adding evidence to context).

## Key Features
- **RAG Pipeline**: ChromaDB-based dense retrieval using `all-MiniLM-L6-v2`.
- **Comparative Evaluation**: Side-by-side analysis of "Base" vs. "RAG" performance.
- **LLM-as-a-Judge**: Automated scoring of generated text using a dedicated judge model (Gemma-4-E4B).

## System Architecture
The pipeline follows a standard RAG workflow optimized for evaluation:
1. **Ingestion**: Wikipedia passages are embedded using `all-MiniLM-L6-v2` and stored in **ChromaDB**.
2. **Querying**: User questions are converted into vectors, the system retrieves the top-5 most similar passages.
3. **Augmentation**: Retrieved passages are injected into the prompt for the generator.
4. **Generation**: **Qwen/Qwen3-4B** generates an answer based on the provided context.
5. **Judging**: **google/gemma-4-E4B-it** compares the output against the ground truth and responds with Yes/No depending on their similarity.

## Evaluation & Results
A core component of this project is the objective measurement of RAG's effectiveness.
- **Methodology**: A separate judge model evaluates every prediction from both "Base" and "RAG" modes.
- **Outputs**: The project generates `res-base.csv`, `res-rag.csv`, and their corresponding judge-scored files.
- **Key Findings**: 
  - **Base Mode (No Retrieval)**: 476/918 correct (~51.9%)
  - **RAG Mode (With Retrieval)**: 714/918 correct (~77.8%)
  - **Improvement**: +238 correct answers (~26.8% percentage points)

## Installation & Usage
### Installation
```bash
uv sync
```

### Running Evaluation
Run the inference for either mode:
```bash
uv run python main.py --mode [base|rag]
```
Run the automated judge:
```bash
uv run python llm-judge.py
```

Run the `result_analysis.ipynb` notebook to compare results (correct answers).

## Tech Stack
- **Models**: Qwen/Qwen3-4B (Generator), google/gemma-4-E4B-it (Judge)
- **Vector Database**: ChromaDB
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Frameworks**: PyTorch, Hugging Face Transformers, Pandas
