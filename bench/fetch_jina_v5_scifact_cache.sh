#!/usr/bin/env bash
# Fetch precomputed jina-embeddings-v5-text-nano embeddings for BEIR/SciFact
# into the layout that bench/eval_beir.py expects:
#
#   bench/.cache/JINA_V5_BEIR_SCIFACT/corpus.npy   (5183, 768) float32
#   bench/.cache/JINA_V5_BEIR_SCIFACT/queries.npy  (300,  768) float32
#   bench/.cache/JINA_V5_BEIR_SCIFACT/meta.json    provenance
#
# Encoding takes ~80 minutes on a CPU container (5183 corpus + 300 queries
# at 1.6 text/s, retrieval adapter, max_length=512). This skips that step.
#
# Source: oaustegard/claude-container-layers releases, tag `jina-v5-nano-scifact`.
#
# Auth:
#   - GH_TOKEN must be set (any classic PAT, no scopes needed beyond public read).
#
# Re-run safe — overwrites the destination.
set -euo pipefail

REPO=${JINA_V5_CACHE_REPO:-oaustegard/claude-container-layers}
TAG=${JINA_V5_CACHE_TAG:-jina-v5-nano-scifact}
CORPUS_ASSET=jina_v5_nano_scifact_corpus.npy
QUERIES_ASSET=jina_v5_nano_scifact_queries.npy
META_ASSET=jina_v5_nano_scifact_meta.json

CACHE_DIR="$(cd "$(dirname "$0")" && pwd)/.cache/JINA_V5_BEIR_SCIFACT"
mkdir -p "$CACHE_DIR"

if [ -z "${GH_TOKEN:-}" ]; then
  echo "error: GH_TOKEN must be set (any classic PAT)." >&2
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
        print(a['id'])
        break
"
}

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

for pair in \
  "$CORPUS_ASSET:$CACHE_DIR/corpus.npy" \
  "$QUERIES_ASSET:$CACHE_DIR/queries.npy" \
  "$META_ASSET:$CACHE_DIR/meta.json"; do
  asset_name="${pair%%:*}"
  dest="${pair##*:}"
  asset_id=$(asset_id_for "$asset_name")
  if [ -z "$asset_id" ]; then
    echo "error: asset $asset_name not found on $REPO@$TAG" >&2
    exit 1
  fi
  echo "downloading $asset_name (asset_id=$asset_id) → $dest"
  download_asset "$asset_id" "$dest" || exit 1
done

python3 - "$CACHE_DIR" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
d = Path(sys.argv[1])
corpus = np.load(d / "corpus.npy", mmap_mode="r")
queries = np.load(d / "queries.npy", mmap_mode="r")
meta = json.loads((d / "meta.json").read_text())
print(f"  corpus:  shape={corpus.shape}, dtype={corpus.dtype}")
print(f"  queries: shape={queries.shape}, dtype={queries.dtype}")
print(f"  meta:    {meta}")
assert corpus.shape == (5183, 768), corpus.shape
assert queries.shape == (300, 768), queries.shape
print("  ok")
PY

echo "done. run: python bench/eval_beir.py --eval-only"
