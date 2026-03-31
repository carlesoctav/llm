#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}" exec python -m jaxformers.dedup_free_args "$@"
