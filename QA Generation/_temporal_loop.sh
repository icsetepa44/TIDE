#!/bin/bash
# Run add_temporal_qa repeatedly until every QA file is in the done-list (call-
# failures stay out of the done-list, so resume retries only them). Low concurrency
# rides out the free-tier per-minute throttling.
cd "$(dirname "$0")"
for i in $(seq 1 10); do
  LEFT=$(python3 -c "
import json, glob, os
done=set(json.load(open('temporal_qa_done.json'))) if os.path.exists('temporal_qa_done.json') else set()
files=[os.path.basename(f) for f in glob.glob('concept_qa/*_QA.json')]
print(sum(1 for f in files if f not in done))
")
  echo "[temporal] iteration $i: $LEFT files still to process"
  if [ "$LEFT" -eq 0 ]; then echo "[temporal] all processed"; break; fi
  python3 add_temporal_qa.py --concurrency 5 >> temporal_run.log 2>&1
  sleep 8
done
echo "[temporal] finished"
