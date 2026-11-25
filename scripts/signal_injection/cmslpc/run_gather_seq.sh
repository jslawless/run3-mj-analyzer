#!/bin/bash
# Sequential pairs-copy + gather for the remaining injection studies.
# Stops at the first study with a failed copy or a nonzero-exit gather job.
cd ~/nobackup/run3-mj-mixer
export X509_USER_PROXY=~/x509up_u59599
for s in M300_x4 M1000_x1 M1000_x4; do
  D=/store/user/jlawless/mixing-inject-26-11-12/$s/pairs
  echo "=== $s copy start $(date)"
  xrdfs root://cmseos.fnal.gov mkdir -p $D || { echo "mkdir failed"; exit 1; }
  for f in inject/pairs_$s/pairs_*.root; do
    xrdcp -s -f "$f" "root://cmseos.fnal.gov/$D/$(basename $f)" || { echo "copy failed: $f"; exit 1; }
  done
  echo "=== $s submit $(date)"
  out=$(condor_submit inject/batch_gather_$s/submit.sub 2>&1); echo "$out" | grep -E "submitted|Attempting"
  cl=$(echo "$out" | sed -n "s/.*submitted to cluster \([0-9]*\).*/\1/p")
  sd=$(echo "$out" | sed -n "s/.*submit jobs to \([^ ]*\)\.*$/\1/p" | sed "s/\.$//")
  [ -n "$cl" ] || { echo "submit failed"; exit 1; }
  echo "$s cluster $cl on $sd" >> inject/gather_clusters.txt
  while [ "$(condor_q -name $sd $cl -af ProcId 2>/dev/null | wc -l)" != 0 ]; do sleep 120; done
  res=$(grep -h "return value" inject/batch_gather_$s/log_${cl}_*.log | sort | uniq -c)
  echo "=== $s done $(date)"; echo "$res"
  n0=$(grep -h "return value 0)" inject/batch_gather_$s/log_${cl}_*.log | wc -l)
  [ "$n0" = 60 ] || { echo "STOP: $s has $n0/60 clean exits"; exit 1; }
done
echo "=== all done $(date)"
