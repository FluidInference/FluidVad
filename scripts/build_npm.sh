#!/usr/bin/env bash
# Build the @fluidinference/fluidvad npm package (npm/dist).
set -euo pipefail
cd "$(dirname "$0")/.."
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web --out-dir npm/dist --release \
  -- --config 'profile.release.opt-level="z"' --config 'profile.release.panic="abort"'
rm -f npm/dist/package.json npm/dist/.gitignore  # npm/package.json is the real manifest
echo "package ready: cd npm && npm publish --access public"
