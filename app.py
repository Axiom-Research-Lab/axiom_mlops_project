import os
import sys

# Add 'src' to the search path so Python can find your package
# This bridges the gap between the root app.py and the src/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from axiom_mlops_project.interface import demo

if __name__ == "__main__":
    # We use 7860 as it's the standard port for Hugging Face Spaces
    # and 0.0.0.0 to ensure it's accessible within the container
    demo.launch(server_name="0.0.0.0", server_port=7860)  # nosec
