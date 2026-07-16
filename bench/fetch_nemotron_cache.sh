#!/usr/bin/env bash
# Fetch precomputed nvidia/Nemotron-3-Embed-1B-BF16 embeddings (2048-dim) and
# the eval inputs for the 1-bit quantization experiment, into the layout that
# bench/eval_nemotron_1bit.py (and the seeds/baselines/latency variants) expect.
#
# By default the scripts read from a scratch dir; point them at the cache with
#   export NEMOTRON_EMB_DIR=bench/.cache/NEMOTRON/emb
#   export NEMOTRON_DATA_DIR=bench/.cache/NEMOTRON/data
# or just copy the files into the paths the scripts already use.
#
# Layout produced:
#   bench/.cache/NEMOTRON/emb/scifact_docs.npy     (1600, 2048) float32
#   bench/.cache/NEMOTRON/emb/scifact_queries.npy  (300,  2048) float32
#   bench/.cache/NEMOTRON/emb/stsb_s1.npy          (1379, 2048) float32
#   bench/.cache/NEMOTRON/emb/stsb_s2.npy          (1379, 2048) float32
#   bench/.cache/NEMOTRON/emb/meta.json
#   bench/.cache/NEMOTRON/data/scifact_subset.json
#   bench/.cache/NEMOTRON/data/stsb_test.json
#
# Encoding these on CPU takes ~1 hour; this skips that step.
#
# Source: oaustegard/claude-container-layers releases, tag `nemotron-3-embed-1b-bf16`.
# Auth: GH_TOKEN must be set (any classic PAT, no scopes needed beyond public read).
# Re-run safe — overwrites the destination.
set -euo pipefail

REPO=${NEMOTRON_CACHE_REPO:-oaustegard/claude-container-layers}
TAG=${NEMOTRON_CACHE_TAG:-nemotron-3-embed-1b-bf16}

CACHE_DIR="$(cd "$(dirname "$0")" && pwd)/.cache/NEMOTRON"
EMB_DIR="$CACHE_DIR/emb"
DATA_DIR="$CACHE_DIR/data"
mkdir -p "$EMB_DIR" "$DATA_DIR"

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
        print(a['id']); break
"
}

download_asset() {
  local asset_id="$1" dest="$2" http
  http=$(curl -sL -o "$dest" -w "%{http_code}" \
    -H "User-Agent: remax-bench" \
    -H "Accept: application/octet-stream" \
    -H "Authorization: token $GH_TOKEN" \
    "https://api.github.com/repos/$REPO/releases/assets/$asset_id")
  if [ "$http" != "200" ]; then
    rm -f "$dest"; echo "error: download failed with HTTP $http for asset $asset_id" >&2; return 1
  fi
}

# asset name on the release  ->  destination path
fetch() {
  local asset="$1" dest="$2" id
  id=$(asset_id_for "$asset")
  if [ -z "$id" ]; then
    echo "error: asset $asset not found on $REPO@$TAG" >&2
    echo "       check https://github.com/$REPO/releases/tag/$TAG" >&2
    exit 1
  fi
  echo "  $asset -> $dest"
  download_asset "$id" "$dest" || exit 1
}

fetch nemotron_scifact_docs.npy     "$EMB_DIR/scifact_docs.npy"
fetch nemotron_scifact_queries.npy  "$EMB_DIR/scifact_queries.npy"
fetch nemotron_stsb_s1.npy          "$EMB_DIR/stsb_s1.npy"
fetch nemotron_stsb_s2.npy          "$EMB_DIR/stsb_s2.npy"
fetch nemotron_meta.json            "$EMB_DIR/meta.json"
fetch nemotron_scifact_subset.json  "$DATA_DIR/scifact_subset.json"
fetch nemotron_stsb_test.json       "$DATA_DIR/stsb_test.json"

python3 - "$EMB_DIR" <<'PY'
import sys, numpy as np
d = sys.argv[1]
for name, shape in [("scifact_docs",(1600,2048)),("scifact_queries",(300,2048)),
                    ("stsb_s1",(1379,2048)),("stsb_s2",(1379,2048))]:
    a = np.load(f"{d}/{name}.npy", mmap_mode="r")
    assert a.shape == shape, f"{name}: expected {shape}, got {a.shape}"
print("  verified all four arrays")
PY

echo "done. Point the eval at the cache:"
echo "  export NEMOTRON_EMB_DIR=$EMB_DIR NEMOTRON_DATA_DIR=$DATA_DIR"
echo "  python3 bench/eval_nemotron_1bit.py"
