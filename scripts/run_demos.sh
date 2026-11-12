#!/usr/bin/env bash
# Regenerates docs/proof/{eval_cli,cli_demo,api_demo}.txt and the evaluation report from real runs.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8011}"
B="http://127.0.0.1:${PORT}"
mkdir -p docs/proof
rm -rf data/index

{ echo '$ raglens eval'; raglens eval 2>&1; } > docs/proof/eval_cli.txt
python scripts/rrf_weight_experiment.py > /dev/null

{
  echo '$ raglens ingest'; raglens ingest 2>&1; echo
  for q in "How does Reciprocal Rank Fusion combine BM25 and dense rankings?" \
           "How do I stop a model from memorising its training examples?" \
           "What is the capital of Australia?"; do
    echo "\$ raglens ask \"$q\""; raglens ask "$q" 2>&1; echo
  done
} > docs/proof/cli_demo.txt

raglens serve --port "$PORT" > /tmp/raglens_api.log 2>&1 &
SERVER=$!
trap 'kill $SERVER' EXIT
for _ in $(seq 1 30); do curl -sf "$B/health" > /dev/null && break; sleep 1; done

post() { curl -s -X POST "$B$1" -H 'content-type: application/json' -d "$2"; }
{
  echo '$ curl /health'; curl -s "$B/health" | python -m json.tool; echo
  echo '$ POST /ask {"question": "What do the k1 and b parameters control in BM25?", "top_k": 3}'
  post /ask '{"question":"What do the k1 and b parameters control in BM25?","top_k":3}' | python -m json.tool; echo
  echo '$ POST /ask {"question": "What is the capital of Australia?"}'
  post /ask '{"question":"What is the capital of Australia?"}' \
    | python -c "import json,sys; d=json.load(sys.stdin); print({k: d[k] for k in ('answer','refused','refusal_reason')})"; echo
  echo '$ POST /ask {"question": "x", "llm_provider": "openai"}   # unknown field'
  curl -s -o /dev/null -w "HTTP %{http_code}\n" -X POST "$B/ask" -H 'content-type: application/json' \
    -d '{"question":"x","llm_provider":"openai"}'; echo
  printf '# Zorblax\n\nThe zorblax protocol rotates quantum widgets every ninety seconds.\n' > /tmp/zorblax.md
  echo '$ curl -F files=@zorblax.md /ingest'; curl -s -F files=@/tmp/zorblax.md "$B/ingest"; echo; echo
  echo '$ POST /ask {"question": "How often does the zorblax protocol rotate widgets?"}'
  post /ask '{"question":"How often does the zorblax protocol rotate widgets?"}' \
    | python -c "import json,sys; d=json.load(sys.stdin); print({k: d[k] for k in ('answer','refused')}); print('cited:', [c['source'] for c in d['citations']])"; echo
  printf 'MZ' > /tmp/notes.exe
  echo '$ curl -F files=@notes.exe /ingest'; curl -s -w "  HTTP %{http_code}\n" -F files=@/tmp/notes.exe "$B/ingest"; echo
  echo '$ POST /eval {"retriever": "hybrid", "split": "test"}'
  post /eval '{"retriever":"hybrid","split":"test"}' | python -m json.tool
} > docs/proof/api_demo.txt 2>&1
echo "Demos written to docs/proof/"
