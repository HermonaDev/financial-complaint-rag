#!/usr/bin/env python3
"""Gradio app with multiple outputs (answer + sources)."""

import gradio as gr

def answer_with_sources(question):
    """Return answer and sample sources."""
    
    question_lower = question.lower()
    
    # Generate answer
    if "credit card" in question_lower:
        answer = "Common credit card complaints include unauthorized charges, billing disputes, and fraud concerns."
        sources = [
            ["Credit Card", "Unauthorized charges", "Customer reported fraudulent card opened in their name..."],
            ["Credit Card", "Billing dispute", "Customer disputed a charge that appeared on statement..."],
            ["Credit Card", "Fraud concern", "Customer couldn't reach fraud department for help..."]
        ]
    elif "savings account" in question_lower:
        answer = "Savings account issues often involve unexpected fees, withdrawal problems, and account access issues."
        sources = [
            ["Savings Account", "Unexpected fees", "Bank charged monthly fee without notification..."],
            ["Savings Account", "Withdrawal problem", "Customer couldn't withdraw funds from ATM..."],
            ["Savings Account", "Account access", "Online banking access disabled unexpectedly..."]
        ]
    else:
        answer = "I can help with questions about credit cards, savings accounts, money transfers, or loans."
        sources = [
            ["General", "Sample", "Ask about specific financial products for detailed analysis..."]
        ]
    
    return answer, sources

# Create Gradio interface with multiple outputs
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# Financial Complaint Analysis Chatbot")
    gr.Markdown("Ask questions about customer complaints across financial products")
    
    with gr.Row():
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g., What issues do customers report with credit cards?",
            lines=3
        )
    
    with gr.Row():
        submit_btn = gr.Button("Ask Question", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")
    
    with gr.Row():
        with gr.Column():
            answer_output = gr.Textbox(
                label="Answer",
                lines=5
            )
    
    with gr.Row():
        sources_output = gr.Dataframe(
            label="Source Complaints",
            headers=["Product", "Issue", "Text Snippet"],
            datatype=["str", "str", "str"],
            row_count=3,  # Fixed number of rows
            col_count=(3, "fixed")  # Fixed number of columns
        )
    
    # Connect buttons to functions
    submit_btn.click(
        fn=answer_with_sources,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )
    
    def clear_all():
        return "", []
    
    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[question_input, sources_output]
    )

if __name__ == "__main__":
    demo.launch()
