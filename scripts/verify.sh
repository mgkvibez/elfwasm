#!/usr/bin/env bash
# One gate, identical locally and in CI: typecheck, tests, and a README count that cannot drift.
set -euo pipefail
cd "$(dirname "$0")/.."

npm run typecheck
npm test
node tools/check-test-count.mjs
