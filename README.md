Here is a professional, high-impact README.md for your GitHub repository. It’s structured to impress recruiters by highlighting the NTCIR-19 competition and the technical complexity of local LLM deployment.

NTCIR-19 R2C2: Movie Knowledge Discovery RAG System
This repository contains a specialized Retrieval-Augmented Generation (RAG) system developed for the NTCIR-19 R2C2 (Answering with Confidence) task. The system is designed to provide factual information from a Wikipedia movie corpus, providing not just answers, but also confidence scores and fact nuggets to ensure "model modesty" and prevent hallucinations.

🚀 Key Features
Local Inference: Runs a quantized Phi-3-mini model locally on a T4 GPU (no API costs).

Modesty-First Design: Implements a custom prompt-engineered scoring system to identify when data is missing from the context.

Fact Extraction: Automatically breaks down complex movie plots into atomic "Nuggets" for better information density.

Scalable Retrieval: Built with LlamaIndex and BGE-small embeddings for high-precision semantic search.

Interactive UI: Deployed via Gradio for real-time testing and demonstration.

🛠️ Tech Stack
Framework: LlamaIndex

LLM: Microsoft Phi-3-mini (Quantized via BitsAndBytes)

Embeddings: BGE-small-en-v1.5

Data Source: Wikipedia Movie Plots (via Hugging Face Datasets)

UI: Gradio

Environment: Python / Google Colab (T4 GPU)

📁 Project Structure
Plaintext
├── movie_data/        # Processed .txt files of movie plots
├── storage/           # Persistent Vector Database (Index)
├── MY_TEAM_AC_1.txt   # Official NTCIR-formatted submission file
└── app.ipynb          # Main application notebook
📊 Evaluation (Modesty Test)
The system was stress-tested with Out-of-Distribution (OOD) queries. When asked about movies not present in the local database (e.g., Deadpool & Wolverine), the system correctly:

Set CONFIDENCE: 0.

Refused to hallucinate facts.

Cited the specific movies it did have access to.

📝 Performance Highlights (Resume Ready)
Quantization: Utilized 4-bit quantization to reduce VRAM footprint by ~60%, enabling high-performance LLM execution on consumer-grade GPUs.

Grounding: Achieved zero hallucinations during "OOD" stress testing by enforcing strict context-only generation rules.

Automation: Developed a automated pipeline to convert raw JSON/CSV corpus data into indexed vector formats.

How to Run
Clone this repository.

Install requirements: pip install llama-index transformers bitsandbytes datasets gradio.

Run the notebook cells to build the index.

Launch the Gradio UI link generated at the end of the script.
