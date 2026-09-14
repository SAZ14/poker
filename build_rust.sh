#!/usr/bin/env bash
# Build the optional Rust acceleration module and drop it next to mccfr.py.
#
# The extension is optional: without it, mccfr.py runs its pure-Python loop and
# every test still passes. With it, MCCFRTrainer transparently uses the native
# Leduc loop and produces bit-identical results.
#
#   ./build_rust.sh          # release build
#   ./build_rust.sh --debug  # faster to compile, much slower to run
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo not found. Install Rust from https://rustup.rs and re-run." >&2
    exit 1
fi

PROFILE_DIR=release
CARGO_ARGS=(--release)
if [[ "${1:-}" == "--debug" ]]; then
    PROFILE_DIR=debug
    CARGO_ARGS=()
fi

cargo build --manifest-path rust/Cargo.toml "${CARGO_ARGS[@]}"

SRC="rust/target/${PROFILE_DIR}/libmccfr_rs.so"
if [[ ! -f "$SRC" ]]; then
    SRC="rust/target/${PROFILE_DIR}/libmccfr_rs.dylib"   # macOS
fi
if [[ ! -f "$SRC" ]]; then
    echo "build produced no shared library under rust/target/${PROFILE_DIR}" >&2
    exit 1
fi

cp "$SRC" ./mccfr_rs.so
echo "built mccfr_rs.so"
python3 -c "import mccfr_rs; print('import ok:', mccfr_rs.__doc__)"
