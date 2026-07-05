#!/usr/bin/env bash
# Download nets/main.nnue from GitHub releases (see fetch-net.ps1).
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
Failed to download main.nnue. The weights are not in git — use a release asset,
copy a compatible FullThreats net (0x6A448AFA) to nets/main.nnue, or set EONEGO_NET at runtime.
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
