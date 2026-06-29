#!/usr/bin/env bash
# Run the full simulation chapter (fig0 + modules A, B, C) and write figures + CSVs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}/.."

python scripts/run_module.py --module all
