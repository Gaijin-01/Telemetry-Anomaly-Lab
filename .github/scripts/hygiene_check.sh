#!/usr/bin/env bash
# Hygiene gate — fails the build on:
#   1. Tracked build artifacts
#   2. Machine-specific paths
#   3. Credential patterns
#   4. Internal agent/workbench markers

set -euo pipefail

FAILED=0
REPORT=""

report_fail() {
    echo "  FAIL: $1"
    FAILED=1
    REPORT+="FAIL: $1\n"
}

# ── 1. Tracked build artifacts ─────────────────────────────────────────────
echo "=== 1. Tracked build artifacts ==="
ARTIFACTS=$(git ls-files | grep -E '(\.whl$|/dist/|/build/|\.egg-info/|\.egg$|/htmlcov/)' || true)
if [ -n "$ARTIFACTS" ]; then
    echo "FAIL: tracked build artifacts found:"
    echo "$ARTIFACTS" | while read -r f; do report_fail "tracked artifact: $f"; done
else
    echo "  PASS: no tracked build artifacts"
fi

# ── 2. Tracked cache artifacts ──────────────────────────────────────────────
echo ""
echo "=== 2. Tracked cache artifacts ==="
CACHE=$(git ls-files | grep -E '(\.ruff_cache|\.mypy_cache|\.pytest_cache|__pycache__|\.hypothesis/)' || true)
if [ -n "$CACHE" ]; then
    echo "FAIL: tracked cache artifacts found:"
    echo "$CACHE" | while read -r f; do report_fail "tracked cache: $f"; done
else
    echo "  PASS: no tracked cache artifacts"
fi

# ── 3. Machine-specific paths ──────────────────────────────────────────────
echo ""
echo "=== 3. Machine-specific paths ==="
MACHINE_PATTERNS=(
    "/mnt/c/Users"
    "/home/sefi"
    "/mnt/c/Users/sefi"
    "C:\\\\Users"
    "\\\\DESKTO"
    "/Users/sefi"
    "/tmp/anomaly-lab"
    "/tmp/ci_"
)
MACHINE_FOUND=""
for pattern in "${MACHINE_PATTERNS[@]}"; do
    FOUND=$(git ls-files -z | xargs -0 grep -lF "$pattern" 2>/dev/null | grep -vE '\.github/scripts/hygiene_check\.sh$|\.github/workflows/ci\.yml$' || true)
    if [ -n "$FOUND" ]; then
        MACHINE_FOUND+="  pattern='$pattern':\n"
        FOUND_FILES=$(echo "$FOUND" | head -10)
        MACHINE_FOUND+="$(echo "$FOUND_FILES" | sed 's/^/    /')\n"
    fi
done
if [ -n "$MACHINE_FOUND" ]; then
    echo "FAIL: machine-specific paths found:"
    echo -e "$MACHINE_FOUND" | head -30
    echo "$MACHINE_FOUND" | while read -r l; do report_fail "machine path: $l"; done
else
    echo "  PASS: no machine-specific paths"
fi

# ── 4. Credential patterns ──────────────────────────────────────────────────
echo ""
echo "=== 4. Credential patterns ==="
CRED_PATTERNS=(
    "ghp_[A-Za-z0-9]\{36,\}"
    "github_pat_[A-Za-z0-9_\-]\{70,\}"
    "xox[baprs]-[A-Za-z0-9\-]\{48,\}"
    "Bearer\s\+[A-Za-z0-9_\-\.]\{20,\}"
    "GITHUB_TOKEN\s*=\s*[A-Za-z0-9_\-]\{20,\}"
    "HuggingFace\s*:\s*hf_[A-Za-z0-9]\{34,\}"
)
CRED_FOUND=""
for pattern in "${CRED_PATTERNS[@]}"; do
    FOUND=$(git ls-files -z | xargs -0 grep -lE "$pattern" 2>/dev/null | grep -vE "(^|/)ci\.yml$|\.github/scripts/hygiene_check\.sh$" || true)
    if [ -n "$FOUND" ]; then
        CRED_FOUND+="  pattern='$pattern': $FOUND\n"
    fi
done
if [ -n "$CRED_FOUND" ]; then
    echo "FAIL: credential patterns found:"
    echo -e "$CRED_FOUND"
    CRED_FILES=$(echo -e "$CRED_FOUND" | grep -oP '(?<=pattern=)[^:]+' | sort -u)
    echo "$CRED_FILES" | while read -r f; do report_fail "credential pattern in: $f"; done
else
    echo "  PASS: no credential patterns"
fi

# ── 5. Agent / workbench markers ───────────────────────────────────────────
echo ""
echo "=== 5. Agent/workbench markers ==="
AGENT_PATTERNS=(
    "@@agent"
    "@@hermes"
    "HERMES_RUNTIME"
    "HANDOFF_CONTEXT"
    "INTERNAL_MARKER"
    "WORKTREE_METADATA"
    "DELEG_BLOB"
    "SEFIROT_INTERNAL"
)
AGENT_FOUND=""
for pattern in "${AGENT_PATTERNS[@]}"; do
    FOUND=$(git ls-files -z | xargs -0 grep -lE "$pattern" 2>/dev/null | grep -vE "(^|/)ci\.yml$|\.github/scripts/hygiene_check\.sh$" || true)
    if [ -n "$FOUND" ]; then
        AGENT_FOUND+="  pattern='$pattern': $FOUND\n"
    fi
done
if [ -n "$AGENT_FOUND" ]; then
    echo "FAIL: agent/workbench markers found:"
    echo -e "$AGENT_FOUND"
    echo -e "$AGENT_FOUND" | while read -r l; do report_fail "agent marker: $l"; done
else
    echo "  PASS: no agent/workbench markers"
fi

# ── 6. Hardcoded user paths in source ──────────────────────────────────────
echo ""
echo "=== 6. Hardcoded user paths in source ==="
USER_PATTERNS=(
    "/home/sefi"
    "/Users/"
    "/mnt/c/Users"
)
USER_FOUND=""
for pattern in "${USER_PATTERNS[@]}"; do
    FOUND=$(git ls-files -z | xargs -0 grep -lE "$pattern" 2>/dev/null | grep -vE '\.github/scripts/hygiene_check\.sh$|\.github/workflows/ci\.yml$' || true)
    if [ -n "$FOUND" ]; then
        USER_FOUND+="  pattern='$pattern': $FOUND\n"
    fi
done
if [ -n "$USER_FOUND" ]; then
    echo "FAIL: hardcoded user paths found:"
    echo -e "$USER_FOUND"
    echo -e "$USER_FOUND" | while read -r l; do report_fail "hardcoded user path: $l"; done
else
    echo "  PASS: no hardcoded user paths"
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Summary ==="
if [ $FAILED -eq 0 ]; then
    echo "ALL HYGIENE CHECKS PASSED"
    exit 0
else
    echo "HYGIENE CHECKS FAILED — see above"
    exit $FAILED
fi
