#!/usr/bin/env bash
# Fetch the BEIR NFCorpus dataset into the layout that
# remax.bench.bm25_sketch expects:
#
#   bench/.cache/NFCorpus/corpus.jsonl       one JSON per doc:   {_id, title, text}
#   bench/.cache/NFCorpus/queries.jsonl      one JSON per query: {_id, text}
#   bench/.cache/NFCorpus/qrels/test.tsv     header + qid\tdocid\tscore rows
#
# Source: the canonical BEIR mirror at TU Darmstadt's UKP lab. Public,
# no auth required. Tiny (~30 MB compressed, ~120 MB unpacked).
#
# After download the SHA-256 of the zip is verified against a pinned
# expected value to catch silent upstream changes. If you regenerate
# the bench after BEIR republishes the zip, update SHA256_EXPECTED
# (or pass --no-verify-sha to opt out for one run).
#
# Re-run safe — re-downloads and re-extracts.

set -euo pipefail

URL=${NFCORPUS_URL:-https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip}
SHA256_EXPECTED=${NFCORPUS_SHA256:-efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b}

CACHE_DIR="$(cd "$(dirname "$0")" && pwd)/.cache/NFCorpus"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$CACHE_DIR"

VERIFY_SHA=1
for arg in "$@"; do
  case "$arg" in
    --no-verify-sha) VERIFY_SHA=0 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
  esac
done

ZIP="$TMP_DIR/nfcorpus.zip"
echo "fetching $URL ..."
# UKP's CDN 403s on the default curl UA — set a browser-shaped UA so the
# request is served. Public dataset, no auth.
curl -fL --retry 3 --retry-delay 2 \
  -A "Mozilla/5.0 (remax-bench/0.0.0)" \
  -o "$ZIP" "$URL"

if [ "$VERIFY_SHA" = "1" ]; then
  echo "verifying sha256 ..."
  ACTUAL=$(sha256sum "$ZIP" | awk '{print $1}')
  if [ "$ACTUAL" != "$SHA256_EXPECTED" ]; then
    echo "error: sha256 mismatch" >&2
    echo "       expected: $SHA256_EXPECTED" >&2
    echo "       actual:   $ACTUAL" >&2
    echo "       (rerun with --no-verify-sha to skip this check, or update the" >&2
    echo "        SHA256_EXPECTED constant if BEIR republished the zip)" >&2
    exit 1
  fi
else
  echo "skipping sha256 verification (--no-verify-sha)"
fi

echo "unpacking ..."
unzip -q -o "$ZIP" -d "$TMP_DIR"

# The BEIR zip extracts a top-level ``nfcorpus/`` directory with the
# three artefacts we need.
SRC="$TMP_DIR/nfcorpus"
if [ ! -d "$SRC" ]; then
  echo "error: expected $SRC after extraction, but the zip layout differs" >&2
  echo "       contents of $TMP_DIR:" >&2
  ls -la "$TMP_DIR" >&2
  exit 1
fi

cp "$SRC/corpus.jsonl"  "$CACHE_DIR/corpus.jsonl"
cp "$SRC/queries.jsonl" "$CACHE_DIR/queries.jsonl"
mkdir -p "$CACHE_DIR/qrels"
cp -r "$SRC/qrels/." "$CACHE_DIR/qrels/"

# Sanity-check the cache. Bail loudly if anything is malformed so the
# bench harness doesn't have to.
python3 - "$CACHE_DIR" <<'PY'
import json, sys
from pathlib import Path

cache = Path(sys.argv[1])
corpus = cache / "corpus.jsonl"
queries = cache / "queries.jsonl"
qrels = cache / "qrels" / "test.tsv"

n_docs = sum(1 for _ in open(corpus, encoding="utf-8") if _.strip())
n_q    = sum(1 for _ in open(queries, encoding="utf-8") if _.strip())
n_rel  = sum(1 for _ in open(qrels, encoding="utf-8") if _.strip()) - 1  # minus header

# Verify the first record of each parses as expected.
with open(corpus, encoding="utf-8") as fh:
    rec = json.loads(next(fh))
    assert "_id" in rec and ("text" in rec or "title" in rec), rec
with open(queries, encoding="utf-8") as fh:
    rec = json.loads(next(fh))
    assert "_id" in rec and "text" in rec, rec
with open(qrels, encoding="utf-8") as fh:
    header = next(fh).rstrip("\n").split("\t")
    assert header[:3] == ["query-id", "corpus-id", "score"], header

print(f"  docs={n_docs}, queries={n_q}, qrels(test)={n_rel}")
print("  ok")
PY

echo "done. run: python bench/run_bm25_sketch.py"
