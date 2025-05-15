#!/bin/bash

WORK_DIR="$(dirname "$(realpath "$0")")/.."
cd "$WORK_DIR"
echo "Running on $(pwd)..."


black *.py pwl_model --exclude=venv
echo "Running black on all Python files..."

isort *.py pwl_model --skip=venv
echo "Running isort on all Python files..."

echo "All checks passed!"