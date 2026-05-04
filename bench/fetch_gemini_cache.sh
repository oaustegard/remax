#!/usr/bin/env bash
# Fetch precomputed Gemini gemini-embedding-001 embeddings into the layout
# remax.bench.datasets expects:
#
#   bench/.cache/GEMINI/embeddings.npy   (10000, 3072) float32
#
# Source: oaustegard/claude-container-layers releases (the same artifact
# remex consumes). 10K NLP-broad papers, Matryoshka-trained, L2-normalized.
#
# Auth and download path mirror fetch_specter2_cache.sh — set GH_TOKEN
# (any classic PAT with default public read) before running.
#
# Re-run safe — overwrites the destination.

set -euo pipefail

REPO=${GEMINI_CACHE_REPO:-oaustegard/claude-container-layers}
TAG=${GEMINI_CACHE_TAG:-gemini-nlp-broad-10k}
ASSET=gemini_nlp_broad.npy

CACHE_DIR="$(cd "$(dirname "$0")" && pwd)/.cache/GEMINI"
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
if arr.ndim != 2 or arr.shape[1] != 3072:
    sys.exit(f"unexpected shape {arr.shape}; expected (*, 3072)")
norms = np.linalg.norm(arr[:100], axis=1)
print(f"  norms[:100] mean={norms.mean():.4f} (expect ~1.0 for L2-normalized)")
print("  ok")
PY

echo "done. run: python bench/sketch_matryoshka.py --dataset GEMINI"
