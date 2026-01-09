import gradio as gr
import os
import sys
import time

# Add src to path to ensure imports work
sys.path.append(os.path.abspath("src"))

from src.retrieval.rag_retriever import RAGRetriever
from src.generation.answer_generator import AnswerGenerator

# Initialize components
print("Initializing RAG components...")
try:
    retriever = RAGRetriever()
    generator = AnswerGenerator()
    print("RAG components initialized.")
except Exception as e:
    print(f"Error initializing RAG components: {e}")
    retriever = None
    generator = None

def format_sources(chunks):
    """
    Formats the retrieved chunks for display in the UI.
    """
    if not chunks:
        return "No sources found."
    
    formatted = ""
    for i, chunk in enumerate(chunks):
        formatted += f"**Source {i+1}** (Score: {chunk.get('score', 0):.4f})\n"
        formatted += f"*Product:* {chunk.get('product', 'N/A')} | *Issue:* {chunk.get('issue', 'N/A')}\n"
        formatted += f"*Text:*\n> {chunk.get('document', '')}\n"
        formatted += "---\n\n"
    return formatted

def rag_chat(query):
    """
    Processing function for the chat interface.
    """
    if not query:
        return "", "Please enter a question."
    
    if not retriever or not generator:
        return "System Error: RAG components not initialized. Please check logs.", ""

    try:
        # 1. Retrieve
        retrieved_chunks = retriever.search(query, k=5)
        
        # 2. Generate
        answer = generator.generate_answer(query, retrieved_chunks)
        
        # 3. Format sources
        sources_text = format_sources(retrieved_chunks)
        
        return answer, sources_text
        
    except Exception as e:
        return f"Error processing request: {str(e)}", ""

# Define Custom CSS
custom_css = """
.container { max-width: 900px; margin: auto; }
.chatbot { height: 400px; overflow-y: auto; }
"""

# Build Gradio Interface
with gr.Blocks(title="Financial Complaint RAG Chatbot") as demo:
    gr.Markdown(
        """
        # 🏦 Financial Complaint Analysis Chatbot
        
        Ask questions about customer complaints related to **Credit Cards, Savings Accounts, Money Transfers, and Personal Loans**.
        This system uses RAG (Retrieval-Augmented Generation) to analyze real-world complaint data.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What are the common issues with credit cards?",
                lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear")
        
    with gr.Row():
        with gr.Column():
            answer_output = gr.Markdown(label="AI Answer", value="**Answer will appear here...**")
            
    with gr.Accordion("📚 Retrieved Source Context (Verify the answer)", open=True):
        sources_output = gr.Markdown(label="Source Chunks")

    # Event handlers
    submit_btn.click(
        fn=rag_chat,
        inputs=[query_input],
        outputs=[answer_output, sources_output]
    )
    
    # Allow pressing Enter to submit
    query_input.submit(
        fn=rag_chat,
        inputs=[query_input],
        outputs=[answer_output, sources_output]
    )
    
    # Client-side clear function (instant, no network request)
    clear_btn.click(
        fn=None,
        inputs=None,
        outputs=[query_input, answer_output, sources_output],
        js="() => ['', '**Answer will appear here...**', '']"
    )

if __name__ == "__main__":
    # Launch the Gradio app
    # Note: Queue is disabled to ensure stability in all environments
    demo.launch(server_name="127.0.0.1", server_port=7867, share=False)
