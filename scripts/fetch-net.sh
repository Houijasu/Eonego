#!/usr/bin/env bash
# Fallback: download nets/main.nnue from GitHub releases when missing (e.g. LFS not pulled).
# Canonical weights are tracked in git under nets/ (see fetch-net.ps1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NET="$ROOT/nets/main.nnue"
VERSION="${1:-latest}"

if [[ -f "$NET" ]]; then
  echo "OK: $NET already present ($(du -h "$NET" | cut -f1))"
  exit 0
fi

mkdir -p "$ROOT/nets"
if [[ "$VERSION" == "latest" ]]; then
  URL="https://github.com/Houijasu/Eonego/releases/latest/download/main.nnue"
else
  TAG="$VERSION"
  [[ "$TAG" == v* ]] || TAG="v$TAG"
  URL="https://github.com/Houijasu/Eonego/releases/download/${TAG}/main.nnue"
fi

echo "Downloading main.nnue from $URL ..."
if ! curl -fsSL -o "$NET" "$URL"; then
  rm -f "$NET"
  cat >&2 <<'EOF'
Failed to download main.nnue. Either:
  git lfs pull   (canonical copy is in nets/main.nnue via Git LFS), or
  use a release asset, copy a compatible FullThreats net (0x6A448AFA) to nets/main.nnue,
  or set EONEGO_NET at runtime.
EOF
  exit 1
fi

SIZE=$(stat -c%s "$NET" 2>/dev/null || stat -f%z "$NET")
if [[ "$SIZE" -lt 1048576 ]]; then
  rm -f "$NET"
  echo "Download looks truncated (< 1 MB)" >&2
  exit 1
fi

echo "OK: $NET ($(du -h "$NET" | cut -f1))"
