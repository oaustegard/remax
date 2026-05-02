#!/usr/bin/env bash
# Fetch precomputed SPECTER2 embeddings into the layout that
# remax.bench.datasets expects:
#
#   bench/.cache/SPECTER2/embeddings.npy   (10000, 768) float32
#
# Source: oaustegard/claude-container-layers releases (the same artifact
# remex/bench/fetch_specter2_cache.sh consumes — single source of truth).
#
# Auth:
#   - The release is public, but downloads route through the GitHub asset
#     API (Accept: application/octet-stream) rather than the CDN redirect,
#     which works more reliably from sandboxed/proxied environments. That
#     path requires a token with public repo read access; any classic PAT
#     works (no scopes needed beyond default public read).
#   - Set GH_TOKEN in the environment before running, or rely on a
#     pre-existing /mnt/project/GitHub.env if you boot Muninn-style.
#
# Re-run safe — overwrites the destination.

set -euo pipefail

REPO=${SPECTER2_CACHE_REPO:-oaustegard/claude-container-layers}
TAG=${SPECTER2_CACHE_TAG:-specter2-nlp-broad-10k}
ASSET=specter2_nlp_broad.npy

CACHE_DIR="$(cd "$(dirname "$0")" && pwd)/.cache/SPECTER2"
mkdir -p "$CACHE_DIR"
DEST="$CACHE_DIR/embeddings.npy"

if [ -z "${GH_TOKEN:-}" ]; then
  echo "error: GH_TOKEN must be set (any classic PAT)." >&2
  echo "       export GH_TOKEN=ghp_..." >&2
  exit 1
fi

API="https://api.github.com/repos/$REPO/releases/tags/$TAG"

echo "looking up $ASSET on $REPO@$TAG..."
ASSET_ID=$(curl -sL \
  -H "User-Agent: remax-bench" \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: token $GH_TOKEN" \
  "$API" \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
for a in d.get('assets', []):
    if a['name'] == '$ASSET':
        print(a['id'])
        break
")

if [ -z "$ASSET_ID" ]; then
  echo "error: asset $ASSET not found on $REPO@$TAG" >&2
  echo "       check release at https://github.com/$REPO/releases/tag/$TAG" >&2
  exit 1
fi

echo "downloading asset_id=$ASSET_ID → $DEST"
HTTP=$(curl -sL -o "$DEST" -w "%{http_code}" \
  -H "User-Agent: remax-bench" \
  -H "Accept: application/octet-stream" \
  -H "Authorization: token $GH_TOKEN" \
  "https://api.github.com/repos/$REPO/releases/assets/$ASSET_ID")

if [ "$HTTP" != "200" ]; then
  echo "error: download failed with HTTP $HTTP" >&2
  rm -f "$DEST"
  exit 1
fi

# Sanity check: verify it's a valid npy of the expected shape
python3 - "$DEST" <<'PY'
import sys
import numpy as np
p = sys.argv[1]
arr = np.load(p, mmap_mode="r")
print(f"  shape={arr.shape}, dtype={arr.dtype}")
if arr.ndim != 2 or arr.shape[1] != 768:
    sys.exit(f"unexpected shape {arr.shape}; expected (*, 768)")
print("  ok")
PY

echo "done. run: python bench/run_baseline.py --datasets SPECTER2"
