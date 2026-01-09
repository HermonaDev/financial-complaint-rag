#!/usr/bin/env python3
"""Financial-themed Gradio app."""

import gradio as gr

def answer_financial_question(question):
    """Simple function to answer financial questions."""
    
    # Simple rule-based answers
    question_lower = question.lower()
    
    if "credit card" in question_lower:
        return "Common credit card complaints include unauthorized charges, billing disputes, and fraud concerns."
    elif "savings account" in question_lower or "bank account" in question_lower:
        return "Savings account issues often involve unexpected fees, withdrawal problems, and account access issues."
    elif "money transfer" in question_lower:
        return "Money transfer complaints typically involve delays, failed transactions, and refund difficulties."
    elif "loan" in question_lower:
        return "Loan complaints usually relate to payment processing, interest rates, and customer service."
    else:
        return "I can help with questions about credit cards, savings accounts, money transfers, or loans. Please ask about one of these topics."

# Create Gradio interface
demo = gr.Interface(
    fn=answer_financial_question,
    inputs=gr.Textbox(
        label="Ask a financial question",
        placeholder="e.g., What are common credit card problems?",
        lines=3  # Makes textbox taller
    ),
    outputs=gr.Textbox(
        label="Answer",
        lines=5  # Makes answer box taller
    ),
    title="Financial Complaint Assistant",
    description="Ask questions about financial complaints (credit cards, savings accounts, money transfers, loans)",
    theme="soft"  # Nice looking theme
)

# Launch with sharing options
if __name__ == "__main__":
    demo.launch(
        share=False,  # Set to True to get a public link
        server_name="127.0.0.1",
        server_port=7860
    )
