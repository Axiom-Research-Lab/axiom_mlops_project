# Meta-task to run all checks
[group('check')]
check: check-format check-lint check-type check-security test

# Check code formatting
[group('check')]
check-format:
    uv run ruff format {{SOURCES}} {{TESTS}}

# Check code quality/linting
[group('check')]
check-lint:
    uv run ruff check --select I --fix {{SOURCES}} {{TESTS}}

# Check type hints
[group('check')]
check-type:
    uv run mypy {{SOURCES}}

# Check for security vulnerabilities
[group('check')]
check-security:
    uv run bandit -r {{SOURCES}}

# Run unit tests with optional arguments
[group('check')]
test args="":
    uv run pytest {{args}} {{TESTS}}
