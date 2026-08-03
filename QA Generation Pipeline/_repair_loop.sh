#!/bin/bash
# Repeatedly re-expand any QA files that still have unexpanded scenarios, until
# none remain (or max iterations). Low concurrency + patient retries ride out the
# free-tier per-minute throttling.
cd "$(dirname "$0")"
for i in $(seq 1 10); do
  LEFT=$(python3 -c "
import json, glob
from pathlib import Path
done=set(json.load(open('expand_scenarios_done.json')))
left=0
for f in glob.glob('concept_qa/*_QA.json'):
    a=json.load(open(f)); b=json.load(open('concept_qa_backup/'+Path(f).name))
    if any(x['type']=='scenario_short_answer' and x.get('scenario')==y.get('scenario') for x,y in zip(b['qa_pairs'],a['qa_pairs'])):
        done.discard(Path(f).name); left+=1
json.dump(sorted(done), open('expand_scenarios_done.json','w'))
print(left)
")
  echo "[repair] iteration $i: $LEFT files still need expansion"
  if [ "$LEFT" -eq 0 ]; then echo "[repair] all done"; break; fi
  python3 expand_scenarios.py --concurrency 2 >> expand_run.log 2>&1
  sleep 10
done
echo "[repair] finished"
