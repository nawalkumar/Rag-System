import os
import traceback
import gradio as gr
from datasets import load_dataset
from typing import List, Dict, TypedDict, Literal
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, END
from langchain_community.retrievers import BM25Retriever

# --- 1. CONFIG & MODELS ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
INDEX_NAME = "movie-index"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
llm = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)

# --- GLOBAL INITIALIZATION (Run Once) ---
print("🚀 Initializing System...")
dataset = load_dataset("AIatMongoDB/embedded_movies", split="train")

docs = []
for item in dataset:
    if item.get('fullplot') and item.get('title'):
        doc = Document(
            page_content=item['fullplot'],
            metadata={"title": item['title'], "year": item.get('year', 0)}
        )
        docs.append(doc)

bm25_retriever = BM25Retriever.from_documents(docs)
print(f"✅ System Ready: {len(docs)} movies indexed.")
# -----------------------------

# --- 2. UPDATED STATE & NODES ---
class RagState(TypedDict):
    query: str
    docs: List[Document]
    output: str
    error: str
    steps: List[str]      # Tracks process for UI
    loop_count: int      # Limit to 3 iterations
    is_relevant: str     # 'yes' or 'no'

def optimize_query_node(state: RagState):
    steps = state.get("steps", [])
    steps.append("🧠 Optimizing query: Identifying potential movie titles...")
    
    prompt = f"""
    You are a search expert. Look at the user query.
    1. If a specific movie is mentioned or clearly described, identify it.
    2. Create a search string that includes the guessed title AND the key plot elements.
    3. IMPORTANT: If you aren't 100% sure of the title, provide a search string that focuses on the PLOT ELEMENTS primarily.
    
    USER QUERY: {state['query']}
    OPTIMIZED SEARCH STRING:"""
    
    optimized_query = llm.invoke(prompt).content.strip()
    steps.append(f"🎯 Targeted Search: {optimized_query}")
    
    return {"query": optimized_query, "steps": steps}

def retrieve_and_rerank_node(state: RagState):
    current_query = state["query"]
    iteration = state.get("loop_count", 0)
    steps = state.get("steps", []) # Get existing steps
    
    if iteration > 0:
        steps.append(f"🔄 Iteration {iteration}: Expanding query...")
        expansion_prompt = f"Rewrite this movie search query to be more effective for a keyword and semantic search. Output only the rewritten query.\nQuery: {current_query}"
        current_query = llm.invoke(expansion_prompt).content
        steps.append(f"🔍 New Search Query: {current_query}")
    else:
        steps.append("📡 Initializing Hybrid Search (BM25 + Pinecone Vector)...")

    try:
        alpha = 0.7 
        vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings, text_key="text")
        
        semantic_results = vectorstore.similarity_search_with_relevance_scores(current_query, k=20)
        bm25_results = bm25_retriever.invoke(current_query, k=20)
        
        combined_scores = {}
        for doc, score in semantic_results:
            combined_scores[doc.page_content] = {"score": score * alpha, "doc": doc}
        
        for i, doc in enumerate(bm25_results):
            rank_score = (20 - i) / 20
            if doc.page_content in combined_scores:
                combined_scores[doc.page_content]["score"] += rank_score * (1 - alpha)
            else:
                combined_scores[doc.page_content] = {"score": rank_score * (1 - alpha), "doc": doc}

        sorted_candidates = [item["doc"] for item in sorted(combined_scores.values(), key=lambda x: x["score"], reverse=True)][:15]
        pairs = [[current_query, d.page_content] for d in sorted_candidates]
        cross_scores = reranker.predict(pairs)
        
        final_ranked = sorted(zip(cross_scores, sorted_candidates), key=lambda x: x[0], reverse=True)
        
        seen, final_docs = set(), []
        for score, doc in final_ranked:
            title = doc.metadata.get('title', 'Unknown')
            if title not in seen:
                doc.metadata['score'] = round(float(score), 3)
                final_docs.append(doc)
                seen.add(title)
            if len(final_docs) >= 5: break

        # CRITICAL: Return 'steps' so they aren't lost
        return {"docs": final_docs, "error": "", "loop_count": iteration, "steps": steps}
    except Exception as e:
        return {"error": f"Retrieval Error: {str(e)}", "steps": steps}
        
def grade_documents_node(state: RagState):
    steps = state.get("steps", [])
    if not state["docs"]:
        steps.append("❌ No documents found. Moving to retry...")
        return {"is_relevant": "no", "loop_count": state["loop_count"] + 1, "steps": steps}
    
    steps.append("⚖️ Grading context quality against query...")
    top_doc_content = state["docs"][0].page_content
    grading_instruction = f"Query: {state['query']}\nDocument: {top_doc_content}\nIs this relevant? Answer YES or NO and reason."
    
    grade_result = llm.invoke(grading_instruction).content.strip()
    
    if grade_result.upper().startswith("YES"):
        steps.append("✅ Context Verified: High relevance match found.")
        return {"is_relevant": "yes", "steps": steps}
    else:
        steps.append(f"⚠️ Context Rejected: {grade_result[:100]}...")
        return {"is_relevant": "no", "loop_count": state["loop_count"] + 1, "steps": steps}

def generate_ntcir_node(state: RagState):
    steps = state.get("steps", [])
    steps.append("Finalizing: Generating NTCIR-19 Report...")
    
    context = "\n\n".join([f"[{i+1}] TITLE: {d.metadata['title']}\nPLOT: {d.page_content}" for i, d in enumerate(state["docs"])])
    
    prompt = f"TASK: NTCIR-19 MOVIE SEARCH REPORT\nCONTEXT:\n{context}\nQUERY: {state['query']}\n" + \
             "REQUIRED FORMAT:\n- **SUMMARY**: 2-3 sentences.\n- **FINDINGS**: Detailed analysis.\n" + \
             "- **SOURCES**: Clean list.\n- **CONFIDENCE_SCORE**: [Score]\n- **MODESTY_SCORE**: [Score]"
    
    table = "### 📊 NTCIR-19 DATA TABLE\n| Rank | Movie Title | Match Score | Match Type |\n| :--- | :--- | :--- | :--- |\n"
    for i, d in enumerate(state["docs"]):
        title = d.metadata.get('title', 'Unknown').split('_')[0]
        score = d.metadata.get('score', '0.0')
        table += f"| {i+1} | {title} | {score} | Hybrid (Vector+BM25) |\n"

    response = llm.invoke(prompt)
    return {"output": table + "\n" + response.content, "steps": steps}

# --- 3. GRAPH BUILDER ---
builder = StateGraph(RagState)

# Add the new Optimizer node
builder.add_node("optimize", optimize_query_node)
builder.add_node("retrieve", retrieve_and_rerank_node)
builder.add_node("grade", grade_documents_node)
builder.add_node("generate", generate_ntcir_node)

# NEW FLOW: Start with optimize -> then retrieve
builder.set_entry_point("optimize")
builder.add_edge("optimize", "retrieve")
builder.add_edge("retrieve", "grade")

# Rest of the logic remains the same...
def decide_to_stop(state):
    if state["is_relevant"] == "yes" or state["loop_count"] >= 3:
        return "generate"
    return "retrieve"

builder.add_conditional_edges("grade", decide_to_stop, {"generate": "generate", "retrieve": "retrieve"})
builder.add_edge("generate", END)
rag_app = builder.compile()
# --- 4. GRADIO UI ---
def run_pipeline(user_query):
    inputs = {
        "query": user_query, 
        "docs": [], 
        "output": "", 
        "error": "", 
        "steps": [], 
        "loop_count": 0, 
        "is_relevant": "no"
    }
    
    current_steps = []
    
    # Use .stream() to get updates after every node execution
    for output in rag_app.stream(inputs):
        # LangGraph returns a dict like {'node_name': {state_updates}}
        for node_name, state_update in output.items():
            if "steps" in state_update:
                # Update our local steps list
                current_steps = state_update["steps"]
                
                # Format the log for the UI
                log_html = "#### 🧠 Agent Process Steps:\n"
                log_html += "\n".join([f"- {step}" for step in current_steps])
                
                # Yield the logs immediately so the user sees progress
                yield f"{log_html}\n\n*Agent is working...*"

            if "output" in state_update:
                final_output = state_update["output"]
                log_html = "#### 🧠 Agent Process Steps:\n"
                log_html += "\n".join([f"- {step}" for step in current_steps])
                
                # Yield the final combined result
                yield f"{log_html}\n\n{final_output}"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🎥 Agentic NTCIR Movie Expert")
    gr.Markdown("Real-time Agentic Loop with BM25 + Semantic + Reranking")
    
    with gr.Row():
        query_input = gr.Textbox(
            label="Search Query", 
            placeholder="e.g., A 2010 movie about dreaming with DiCaprio...",
            scale=4
        )
        run_btn = gr.Button("🚀 RUN AGENT", variant="primary", scale=1)
    
    # This Markdown component will now update in real-time
    output_display = gr.Markdown("The agent's thought process will appear here...")
    
    # Note: No changes needed to the click function, Gradio handles 'yield' automatically
    run_btn.click(fn=run_pipeline, inputs=query_input, outputs=output_display)

demo.launch()

