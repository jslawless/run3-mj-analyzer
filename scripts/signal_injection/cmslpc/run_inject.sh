#!/bin/bash
# Signal-injection mu-scan inputs (analyzer a1d204a). No pSignal cut, mass
# asymmetry < 0.1, sqrt(xs_weight). First the non-injected August mixed sample
# (reference for mu = 0), then the four studies split by pair type (signal
# file_id >= 5482 in all four file tables). One pass at a time.
# Run on cmslpc; outputs go to $1 (default the directory used for the results,
# ~/nobackup/run3-mj-analyzer/inject_26_11_23).
set -u
export PYTHONUNBUFFERED=1
OUT=${1:-~/nobackup/run3-mj-analyzer/inject_26_11_23}
mkdir -p "$OUT" && cd "$OUT"
A=~/nobackup/run3-mj-analyzer
EOSD=root://cmseos.fnal.gov//store/group/lpcmultijets/johnny
COMMON="--psignal-cut -1 --mass-asym-cut 0.1 --weight-power 0.5 --no-progress"
files() { for i in $(seq -f %05g 0 90); do echo $EOSD/$1/evaluated_stitched_$i.root; done; }

echo "noninjected start $(date)"
$A/pkg-env/bin/python $A/scripts/make_histograms.py $(files evaluated_mixed_26_11_12) \
  $COMMON -o hists_mixed_noninjected.root > hists_mixed_noninjected.log 2>&1
echo "noninjected exit $? $(date)"

for s in M300_x1 M300_x4 M1000_x1 M1000_x4; do
  echo "$s start $(date)"
  $A/pkg-env/bin/python $A/scripts/make_histograms.py $(files evaluated_inject_26_11_13/$s) \
    --signal-file-id-min 5482 $COMMON -o hists_inject_$s.root > hists_inject_$s.log 2>&1
  echo "$s exit $? $(date)"
done
echo "all done $(date)"
