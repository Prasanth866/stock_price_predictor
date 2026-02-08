#!/usr/bin/env bash
set -euo pipefail

# Run the CLI prediction pipeline. Additional args are forwarded to python -m src.main
poetry run python -m src.main "$@"
