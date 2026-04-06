# Main configuration file for Just.
set dotenv-load := true # Automatically load environment variables

# VARIABLES: Define global constants
SOURCES := "src"
TESTS := "tests"

# DEFAULTS: Display a list of available tasks.
default:
    @just --list

# IMPORTS: Modularize tasks
import 'tasks/check.just'

# PROJECT TASKS
run:
    uv run src/main.py

clean:
    rm -rf .pytest_cache .mypy_cache .ruff_cache
    rm -f axiom_engine.log
    rm -f models/*.joblib
