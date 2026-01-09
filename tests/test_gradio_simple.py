#!/usr/bin/env python3
"""Simple Gradio test to understand basics."""

import gradio as gr

def greet(name):
    """Simple function that returns a greeting."""
    return f"Hello {name}! Welcome to the Financial Complaint Chatbot."

# Create Gradio interface
demo = gr.Interface(
    fn=greet,  # Function to call
    inputs=gr.Textbox(label="Enter your name", placeholder="Type your name here..."),
    outputs=gr.Textbox(label="Greeting"),
    title="Simple Gradio Test",
    description="This is a basic Gradio interface to understand how it works."
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
