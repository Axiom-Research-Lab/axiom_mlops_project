# 1. Base image
FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# 2. Set working directory
WORKDIR /app

# 3. Copy the project assets (Data and Models)
# This ensures the container has the CSV and any saved models
COPY data/ ./data/
COPY models/ ./models/

# 4. Copy the pre-built Python wheel
COPY dist/*.whl .

# 5. Install the wheel
RUN uv pip install --system *.whl

# 6. Run the command
CMD ["axiom-mlops"]