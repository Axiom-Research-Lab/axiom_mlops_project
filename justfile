# Axiom MLOps Project: Strategy Runner

# Run the full training pipeline
train:
    python src/axiom_mlops_project/main.py

# --- Section 7.0: Reproducibility & Deterministic Packaging ---

# 1. Generate dependency constraints with hashes
# This now includes 'setuptools' and 'wheel' because we added them as build-deps.
package-constraints constraints="constraints.txt":
    uv pip compile pyproject.toml --generate-hashes --output-file={{constraints}}

# 2. Build a deterministic, hash-verified python wheel
# We use the verified environment to ensure bit-for-bit identity.
[group('package')]
package-build constraints="constraints.txt": package-constraints
    uv build --build-constraint={{constraints}} --require-hashes --wheel

# 3. Clean up build artifacts
[group('package')]
clean-build:
    rm -rf dist/ build/ *.egg-info
    rm -f constraints.txt
