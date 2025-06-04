"""
EulerMCP - Learning Knowledge Graph
Main application file for Hugging Face Spaces deployment.
"""

import os
import sys
import gradio as gr

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.euler_mcp.gradio_app import create_gradio_interface

# Create the Gradio interface
app = create_gradio_interface()

# Launch the app
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )