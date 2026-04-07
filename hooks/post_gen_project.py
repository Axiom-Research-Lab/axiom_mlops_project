import os

print("Running Axiom Lab post-generation hook...")

# Example logic: Remove MLflow config if Weights & Biases is selected
# In a full template, this Jinja2 logic evaluates during generation:
REMOVE_PATHS = [
    "{% if cookiecutter.tracking_tool != 'MLflow' %}mlflow_config.py{% endif %}",
]

for path in REMOVE_PATHS:
    path = path.strip()
    if path and os.path.exists(path):
        os.remove(path)
        print(f"Removed unused config: {path}")

print("Project successfully initialized!")
