#!/usr/bin/env bash
# Fetch precomputed SPECTER2 embeddings (and their source texts) into the
# layout that remax.bench.datasets expects:
#
#   bench/.cache/SPECTER2/embeddings.npy   (10000, 768) float32
#   bench/.cache/SPECTER2/texts.json       list[str] of "title [SEP] abstract"
#
# Texts are needed by stage-2 cross-encoder rerank experiments (issue #20);
# the baseline harness only needs the .npy. Texts download fails soft — if
# the asset isn't on the release the script still completes successfully
# with embeddings only.
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
EMB_ASSET=specter2_nlp_broad.npy
TEXTS_ASSET=specter2_nlp_broad_texts.json

CACHE_DIR="$(cd "$(dirname "$0")" && pwd)/.cache/SPECTER2"
mkdir -p "$CACHE_DIR"
EMB_DEST="$CACHE_DIR/embeddings.npy"
TEXTS_DEST="$CACHE_DIR/texts.json"

if [ -z "${GH_TOKEN:-}" ]; then
  echo "error: GH_TOKEN must be set (any classic PAT)." >&2
  echo "       export GH_TOKEN=ghp_..." >&2
  exit 1
fi

API="https://api.github.com/repos/$REPO/releases/tags/$TAG"

echo "fetching release manifest for $REPO@$TAG..."
MANIFEST=$(curl -sL \
  -H "User-Agent: remax-bench" \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: token $GH_TOKEN" \
  "$API")

# Resolve an asset id by name from the cached manifest. Empty string → not
# found (caller decides whether that's fatal).
asset_id_for() {
  local name="$1"
  printf '%s' "$MANIFEST" | python3 -c "
import json, sys
name = '$name'
try:
    d = json.loads(sys.stdin.read())
except Exception:
    sys.exit(0)
for a in d.get('assets', []):
    if a['name'] == name:
        print(a['id'])
        break
"
}

# Download asset $1 (id) to $2 (dest path). Returns 0 on HTTP 200.
download_asset() {
  local asset_id="$1"
  local dest="$2"
  local http
  http=$(curl -sL -o "$dest" -w "%{http_code}" \
    -H "User-Agent: remax-bench" \
    -H "Accept: application/octet-stream" \
    -H "Authorization: token $GH_TOKEN" \
    "https://api.github.com/repos/$REPO/releases/assets/$asset_id")
  if [ "$http" != "200" ]; then
    rm -f "$dest"
    echo "error: download failed with HTTP $http" >&2
    return 1
  fi
}

# --- embeddings (required) ---
EMB_ID=$(asset_id_for "$EMB_ASSET")
if [ -z "$EMB_ID" ]; then
  echo "error: asset $EMB_ASSET not found on $REPO@$TAG" >&2
  echo "       check release at https://github.com/$REPO/releases/tag/$TAG" >&2
  exit 1
fi

echo "downloading $EMB_ASSET (asset_id=$EMB_ID) → $EMB_DEST"
download_asset "$EMB_ID" "$EMB_DEST" || exit 1

python3 - "$EMB_DEST" <<'PY'
import sys
import numpy as np
p = sys.argv[1]
arr = np.load(p, mmap_mode="r")
print(f"  shape={arr.shape}, dtype={arr.dtype}")
if arr.ndim != 2 or arr.shape[1] != 768:
    sys.exit(f"unexpected shape {arr.shape}; expected (*, 768)")
print("  ok")
PY

# --- texts (optional, soft-fail) ---
TEXTS_ID=$(asset_id_for "$TEXTS_ASSET")
if [ -z "$TEXTS_ID" ]; then
  echo "note: $TEXTS_ASSET not on release; skipping texts (rerank experiment will be unavailable)."
else
  echo "downloading $TEXTS_ASSET (asset_id=$TEXTS_ID) → $TEXTS_DEST"
  if download_asset "$TEXTS_ID" "$TEXTS_DEST"; then
    python3 - "$TEXTS_DEST" <<'PY'
import json, sys
p = sys.argv[1]
with open(p) as f:
    items = json.load(f)
print(f"  texts={len(items)}, sample={items[0][:80]!r}...")
if not isinstance(items, list) or not all(isinstance(t, str) for t in items):
    sys.exit("texts.json must be a list of strings")
print("  ok")
PY
  else
    echo "warning: texts download failed; continuing with embeddings only." >&2
  fi
fi

echo "done. run: python bench/run_baseline.py --datasets SPECTER2"
echo "      or: python bench/run_rerank.py   (cross-encoder stage-2 experiment, needs texts.json)"
