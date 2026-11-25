#!/bin/zsh
# Nominal-strength mu scan of the four injection studies, from the
# make_histograms.py outputs of cmslpc/run_inject.sh. Writes into the work
# directory (default ~/analysis/histograms/inject_26_11_23, or $INJECT_WORKDIR):
#   plots/<study>/                 decompositions, scans, summaries (default mu grid)
#   plots/<study>/log/             log-y scans + non-injected comparison (+ x1 vs x4 SQ)
#   plots/<study>/mu0_1_3_log/     mu = 0, 1, 3 only
#   plots/<study>/mu0_1_3_100_log/ mu = 0, 1, 3, 100 only
#   plots/mu0_1_3*_log_all_studies.pdf, plots_<study>.txt (printed tables)
# x4 studies run at mu/4 (--mu-scale 0.25) so every label is the x1-equivalent mu.
set -e
HERE=${0:A:h}
ANALYZER=${HERE:h:h}
PY=${PY:-$ANALYZER/pkg-env/bin/python}
S=$ANALYZER/scripts/plot_injection_scan.py
cd ${INJECT_WORKDIR:-~/analysis/histograms/inject_26_11_23}

STUDIES=(M300_x1 M300_x4 M1000_x1 M1000_x4)
OBS_ALL=(ht leadjet_pt mass_PairNetV0_9 mass_PairnetV1 mass_PairNetInRepo mass_PairNetGivenByLarry)
REF=(--reference hists_mixed_noninjected.root)

for st in $STUDIES; do
  m=${st%_*}; x=${st#*_}
  sc=1; extra=()
  if [[ $x == x4 ]]; then
    sc=0.25; extra=(--compare-sq hists_inject_${m}_x1.root --compare-label "$m x1")
  fi
  # 1. default mu grid, linear y: decompositions, scans, summaries, table
  $PY $S hists_inject_$st.root --label "$m $x" --mu-scale $sc --rebin 2 \
      $REF $extra -o plots/$st > plots_$st.txt 2>&1
  # 2. the same scans with log y (kept: scan_*, noninj_*, sqcompare_*)
  D=plots/$st/log
  $PY $S hists_inject_$st.root --label "$m $x" --mu-scale $sc --rebin 2 --log \
      $REF $extra -o $D > /dev/null
  rm -f $D/decomp_* $D/summary_*
  # 3. short mu lists, no highlight
  for mus in "0 1 3" "0 1 3 100"; do
    D=plots/$st/mu${mus// /_}_log
    $PY $S hists_inject_$st.root --label "$m $x" --mu-scale $sc --rebin 2 --log \
        --no-highlight --mu ${=mus} -o $D > /dev/null
    rm -f $D/decomp_* $D/summary_* $D/scan_table.json
  done
done

# one PDF per short-mu set, ordered by observable, then study
for set in mu0_1_3_log mu0_1_3_100_log; do
  FILES=()
  for obs in $OBS_ALL; do
    for st in $STUDIES; do FILES+=(plots/$st/$set/scan_$obs.pdf); done
  done
  pdfunite $FILES plots/${set}_all_studies.pdf
done
