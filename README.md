# 🚀 Hybrid RAG Pipeline: BM25 + FAISS Vector Search + Cross-Encoder Reranking

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange.svg)](https://www.sbert.net/)
[![FAISS](https://img.shields.io/badge/FAISS-VectorDB-green.svg)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/Rank--BM25-Lexical--Search-red.svg)](https://pypi.org/project/rank-bm25/)
[![Cross-Encoder](https://img.shields.io/badge/Cross--Encoder-MS--MARCO-purple.svg)](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
[![Groq](https://img.shields.io/badge/Groq-Llama--3.3--70B-yellow.svg)](https://groq.com/)

An enterprise-grade, two-stage **Hybrid Retrieval-Augmented Generation (Hybrid RAG)** system engineered to overcome the critical limitations of standard dense vector search. By combining **BM25 sparse lexical search** with **FAISS dense vector embeddings** and a **Cross-Encoder reranking model**, this system captures both exact keyword matches and deep semantic intent, maximizing context precision while eliminating LLM hallucinations.

---

## 📌 Executive Summary & Business Usecase

### The Challenge with Single-Vector RAG
Traditional RAG architectures rely exclusively on dense vector similarity (e.g., Cosine or $L2$ Euclidean distance over embeddings). While vector search excels at broad semantic understanding, it frequently fails in production enterprise applications due to:
* **Exact Matching & Vocabulary Gaps**: Fails on specific policy codes, employee IDs, legal clauses, acronyms, or precise numeric constraints (e.g., *"20 days annual leave"*, *"Policy 26-A"*).
* **Semantic Noise**: Retrieves semantically adjacent but contextually incorrect information.
* **Suboptimal Context Order**: Top-$K$ vector matches often place less relevant documents at the top of the LLM context window.

### The Hybrid RAG Solution
This project implements an **Enterprise Knowledge Base & Policy Search Engine** using a two-stage retrieval methodology:
1. **Stage 1 (Hybrid Candidate Retrieval)**: Parallel execution of **BM25Okapi** (exact lexical matching) and **FAISS Vector Indexing** (dense semantic search), merging high-recall document candidates.
2. **Stage 2 (Precision Reranking)**: A transformer-based **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)** evaluates deep joint query-document cross-attention, reordering candidates so only top-relevance context enters the prompt.
3. **Stage 3 (Grounded Synthesis)**: Re-ranked context is passed to **Llama 3.3 70B** (via Groq API) under zero-temperature constraints for deterministic, hallucination-free generation.

---

## ⚙️ High-Level System Architecture

```
                               ┌───────────────────────────┐
                               │       User Query          │
                               └─────────────┬─────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │                                           │
                       ▼                                           ▼
          ┌──────────────────────────┐                ┌──────────────────────────┐
          │   Dense Vector Search    │                │   Sparse Lexical Search  │
          │   SentenceTransformers   │                │        BM25Okapi         │
          │   + FAISS Index (L2)     │                │   (Tokenized Matching)   │
          └────────────┬─────────────┘                └────────────┬─────────────┘
                       │ Top-K Candidates                          │ Top-K Candidates
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Candidate Merge & Dedup   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Cross-Encoder Reranker   │
                               │  ms-marco-MiniLM-L-6-v2   │
                               └─────────────┬─────────────┘
                                             │ Top-K Re-ranked Context
                                             ▼
                               ┌───────────────────────────┐
                               │     Prompt Synthesis      │
                               │ (Strict Grounding Rules)  │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │    Llama 3.3 70B (Groq)   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Final Grounded Answer     │
                               └───────────────────────────┘
```

---

## 📖 Important Terminology & Core Concepts

Key concepts and algorithms implemented in this project:

### 1. Retrieval-Augmented Generation (RAG)
* **Concept**: An AI architecture that connects Large Language Models to external knowledge bases. It retrieves relevant contextual document blocks before passing the prompt to the generator LLM.
* **Impact**: Eliminates model knowledge cutoffs, lowers hallucination rates, and enables verifiable source attribution.

### 2. BM25 (Best Matching 25 / Lexical Search)
* **Concept**: A state-of-the-art term-frequency/inverse-document-frequency ($TF\text{-}IDF$) ranking algorithm that calculates document relevance based on exact keyword occurrences, term saturation, and document length normalization.
* **Impact**: Ensures exact matching for numbers, policy codes, names, and domain-specific terminology that dense embeddings often obscure.

### 3. Dense Embeddings & FAISS Vector Indexing
* **Concept**: Dense embedding models map text into high-dimensional vector spaces ($D=384$ with `all-MiniLM-L6-v2`), representing conceptual intent. **FAISS** (Facebook AI Similarity Search) indexes these vectors for near-instant similarity lookup using Euclidean ($L2$) distance.
* **Impact**: Captures conceptual and semantic intent even when query and document use completely different vocabulary.

### 4. Hybrid Search / Fusion Retrieval
* **Concept**: The integration of sparse (BM25) and dense (FAISS) search results into a unified candidate pool.
* **Impact**: Combines the precision of exact keyword matching with the recall of deep semantic search.

### 5. Bi-Encoder vs. Cross-Encoder Architecture
* **Bi-Encoder**: Encodes queries and documents independently into vector embeddings. Fast for high-scale indexing, but lacks token-level cross-attention.
* **Cross-Encoder**: Processes query and document strings simultaneously through transformer cross-attention layers. Provides deep relevance scoring, but is computationally heavier.
* **Impact**: Deploying the Cross-Encoder as a **Stage 2 Reranker** on candidate items delivers peak precision with minimal latency.

### 6. Context Precision, Recall & Hallucination Mitigation
* **Context Precision**: The ratio of retrieved context chunks that are genuinely useful to the query.
* **Context Recall**: The proportion of all necessary facts retrieved from the corpus.
* **Hallucination Mitigation**: Enforcing strict system prompt boundaries and zero temperature ($T=0.0$) so the LLM relies solely on retrieved facts.

---

## 📊 Performance Benchmark: Standard RAG vs. Hybrid RAG

Evaluating the pipeline on complex HR Policy queries highlights the clear edge of **Hybrid Retrieval + Reranking** over conventional single-vector search.

### Case Study Query: *"Tell me about the leave policies"*

| Metric / Aspect | Standard Dense RAG | Hybrid RAG + Cross-Encoder Reranker |
| :--- | :--- | :--- |
| **Retrieval Strategy** | Single-vector FAISS similarity ($K=5$) | BM25 + FAISS candidate fusion $\rightarrow$ Cross-Encoder reranking |
| **Retrieved Context** | Retained noise (included WFH policy & medical coverage) | Top 3 exact policy matches (Annual Leave, Maternity/Paternity, Probation Rule) |
| **Context Relevance** | Moderate (semantic overlap only) | High (exact keyword & policy alignment) |
| **Answer Completeness** | Missed probation prerequisites | **Complete & accurate** (included 6-month probation constraint) |

### Sample Output Breakdown

```text
==================================================
STANDARD DENSE RAG RESPONSE:
==================================================
The company offers the following leave policies:
1. Paid annual leave: 20 days per year.
2. Maternity leave: 26 weeks.
3. Paternity leave: 15 days.

==================================================
HYBRID RAG + CROSS-ENCODER RESPONSE:
==================================================
The company offers the following leave policies:
1. Annual leave: 20 days of paid leave per year.
2. Maternity leave: 26 weeks.
3. Paternity leave: 15 days.

Note that new employees must complete a six-month probation period before these policies apply.
```

---

## 🛠️ Tech Stack & Key Technologies

| Layer | Technology | Primary Function |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core execution environment |
| **Dense Embeddings** | `SentenceTransformers` (`all-MiniLM-L6-v2`) | High-efficiency 384-dimensional vector embedding |
| **Vector DB** | `FAISS` (`IndexFlatL2`) | In-memory similarity search over vector spaces |
| **Lexical Search** | `Rank-BM25` (`BM25Okapi`) | Probabilistic keyword ranking engine |
| **Reranking Engine** | `Cross-Encoder` (`ms-marco-MiniLM-L-6-v2`) | Deep transformer cross-attention relevance scoring |
| **LLM Inference** | `Groq` API (`llama-3.3-70b-versatile`) | LPU-accelerated generation for grounded answers |

---

## 🚀 Quick Start & Execution

### 1. Requirements & Setup
```bash
git clone https://github.com/your-username/bm25_hybrid_rag.git
cd bm25_hybrid_rag
pip install sentence-transformers transformers faiss-cpu rank-bm25 torch openai
```

### 2. Configure Groq API Key
Obtain an API key from [Groq Console](https://console.groq.com/) and set the environment variable:

* **Linux / macOS**: `export GROQ_API_KEY="your_api_key"`
* **Windows (CMD)**: `set GROQ_API_KEY=your_api_key`

### 3. Run Notebook
Open and run `hybrid_rag.ipynb` in your preferred Jupyter environment:
```bash
jupyter notebook hybrid_rag.ipynb
```

---

## 📈 Enterprise Production Roadmap

Future architectural enhancements for multi-tenant enterprise deployment:
* [ ] **Reciprocal Rank Fusion (RRF)**: Algorithmic rank aggregation over sparse and dense candidate ranks.
* [ ] **Vector Database Scaling**: Upgrade to managed vector stores like **Qdrant**, **Milvus**, or **ChromaDB** with HNSW indexing.
* [ ] **Automated RAG Metrics**: Continuous evaluation via **Ragas** / **TruLens** (measuring Faithfulness, Answer Relevance, and Context Recall).
* [ ] **Metadata Filtering**: Support role-based access control (RBAC) and document filtering during retrieval.

---

## 👨‍💻 Author & Project Credits

Built as a benchmark project in **Information Retrieval (IR)** and **LLM Systems Engineering**.

* **GitHub**: [@your-username](https://github.com/)
* **LinkedIn**: [Your Profile](https://linkedin.com/in/)
* **Domain Focus**: Information Retrieval, Hybrid RAG, Vector Search, AI Systems Engineering

---
*If you find this project valuable, feel free to give it a ⭐️ star on GitHub!*
