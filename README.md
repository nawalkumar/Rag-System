# Retrieval-Augmented Generation System using Phi-3

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for answering user queries using custom documents.
It combines **vector search with a Large Language Model** to generate accurate and context-aware responses.

The system preprocesses documents, splits them into chunks, converts them into embeddings, stores them in a vector database, and retrieves the most relevant chunks to answer user queries.

---

## Architecture

```
Documents
   ↓
Preprocessing
   ↓
Chunking
   ↓
Embedding Generation
   ↓
Vector Database
   ↓
User Query
   ↓
Query Embedding
   ↓
Similarity Search
   ↓
Relevant Chunks Retrieved
   ↓
Phi-3 Language Model
   ↓
Generated Response
```

---

## Technologies Used

* **Microsoft Phi-3** – Large Language Model for response generation
* **LlamaIndex** – Framework for building RAG pipelines
* **BGE-Small Encoder** – Text embedding model
* **Vector Database** – Stores chunk embeddings for similarity search
* **Python** – Implementation language

---

## Key Components

### 1. Data Preprocessing

Raw documents are cleaned before processing to remove unwanted characters, HTML tags, and formatting issues.

### 2. Chunking

Large documents are split into smaller chunks to improve retrieval efficiency.

Example:

```
Document:
Artificial Intelligence is transforming industries...

Chunks:
Chunk 1 → Artificial Intelligence is transforming industries
Chunk 2 → AI helps automate complex tasks
Chunk 3 → Machine learning improves decision making
```

Chunking improves search accuracy and allows the model to process large datasets efficiently.

---

### 3. Embedding Generation

Each chunk is converted into a **vector representation** using the **BGE-Small embedding model**.

Example:

```
Text: "AI helps automate tasks"

Vector:
[0.231, -0.112, 0.876, ...]
```

These vectors capture the semantic meaning of the text.

---

### 4. Vector Database

The system stores:

* Chunk text
* Corresponding embedding vectors

This enables fast similarity search during query time.

Example record:

```
{
  chunk_id: 12,
  text: "AI helps automate tasks",
  embedding: [0.231, -0.112, 0.876 ...]
}
```

---

### 5. Query Processing

When a user submits a query:

1. The query is converted into an embedding vector.
2. The vector database performs **similarity search**.
3. The most relevant chunks are retrieved.

---

### 6. Response Generation

The retrieved chunks are provided as context to the **Phi-3 language model**, which generates the final response.

Example:

```
User Query:
What is Artificial Intelligence?

Retrieved Context:
Artificial intelligence is the simulation of human intelligence...

Generated Answer:
Artificial Intelligence (AI) refers to the simulation of human intelligence
in machines designed to think and learn like humans.
```

---

## Advantages of RAG

* Uses **custom knowledge sources**
* Improves **accuracy and relevance**
* Reduces **hallucination in LLM responses**
* Enables **domain-specific question answering**

---

## Project Workflow

1. Load and preprocess documents
2. Split documents into chunks
3. Generate embeddings for each chunk
4. Store embeddings in a vector database
5. Accept user queries
6. Convert queries to embeddings
7. Retrieve similar chunks
8. Pass context to Phi-3 model
9. Generate final response

---

## Applications

* Document Question Answering
* Knowledge Base Search
* AI Chatbots
* Enterprise Document Retrieval
* Research Assistance Systems

---

## Future Improvements

* Add advanced vector databases (FAISS / Pinecone)
* Implement hybrid search (keyword + vector search)
* Improve chunking strategies
* Add streaming responses
* Deploy as an API or web application

---

## Author

Naval Kumar

B.Tech Computer Science and Engineering
Interested in AI Systems, Machine Learning, and Software Development.
