#!/bin/bash
cd ~/nobackup/run3-mj-mixer
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
source pkg-env/bin/activate
for s in M300_x1 M1000_x1; do
  echo "=== $s start $(date)"
  /usr/bin/time -f "$s: %e s, peak %M KB" run3-mj-match inject/index_$s -t inject/file_table_$s.json -o inject/pairs_$s \
    --max-distance 0.5 --pt-tolerance 0.10 --seed 42 --pairs-per-chunk 100000 > inject/match_$s.log 2>&1
  echo "=== $s exit $? $(date)"
  tail -2 inject/match_$s.log
done
