# This is a Jinja2 template file demonstrating Cookiecutter injection.
# When the template is generated, Jinja2 will replace the bracketed variables.

PROJECT_NAME = "{{ cookiecutter.project_name }}"
AUTHOR = "{{ cookiecutter.author_name }}"
VERSION = "{{ cookiecutter.version }}"


def print_project_metadata():
    """Prints the injected MLOps project metadata."""
    print(f"Initializing {PROJECT_NAME} (v{VERSION}) created by {AUTHOR}.")
