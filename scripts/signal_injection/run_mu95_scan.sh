#!/bin/zsh
# Injection scan at the CMS-EXO-21-004 observed limit (mu95obs), its 10x and
# 100x, and the theory cross section (mu = 1). Plots go to the work directory
# (default ~/analysis/histograms/inject_26_11_23, or $INJECT_WORKDIR), the
# mu95 table next to this script.
#   mu95obs from excluded_xs/mu95_table.py: M300 3.39e-3, M1000 3.00e-2
set -e
HERE=${0:A:h}
ANALYZER=${HERE:h:h}
PY=${PY:-$ANALYZER/pkg-env/bin/python}
S=$ANALYZER/scripts/plot_injection_scan.py
STUDIES=(M300_x1 M300_x4 M1000_x1 M1000_x4)
typeset -A MU95=(M300 0.00339 M1000 0.0300)

$PY $HERE/excluded_xs/mu95_table.py > $HERE/excluded_xs/mu95_table.txt
cd ${INJECT_WORKDIR:-~/analysis/histograms/inject_26_11_23}

for st in $STUDIES; do
  m=${st%_*}; x=${st#*_}
  sc=1; [[ $x == x4 ]] && sc=0.25          # x4: apply mu/4 to match x1
  D=plots/$st/mu95obs_log
  $PY $S hists_inject_$st.root --label "$m $x" --mu-scale $sc --rebin 2 \
      --log --mu95 $MU95[$m] -o $D > plots_mu95obs_$st.txt
  rm -f $D/decomp_* $D/summary_sigfrac*
done

# M300 x1 zoomed to 200-600 (HT left out: HT > 550 leaves it nearly empty)
ZOBS=(leadjet_pt mass_PairNetV0_9 mass_PairnetV1 mass_PairNetInRepo mass_PairNetGivenByLarry)
D=plots/M300_x1/mu95obs_log_zoom200_600
$PY $S hists_inject_M300_x1.root --label "M300 x1" --rebin 2 --log \
    --mu95 $MU95[M300] --xlim 200 600 --obs $ZOBS -o $D > /dev/null
rm -f $D/decomp_* $D/summary_* $D/scan_table.json

# one PDF per set, ordered by observable, then study
FILES=()
for obs in ht leadjet_pt mass_PairNetV0_9 mass_PairnetV1 mass_PairNetInRepo mass_PairNetGivenByLarry; do
  for st in $STUDIES; do FILES+=(plots/$st/mu95obs_log/scan_$obs.pdf); done
done
for st in $STUDIES; do FILES+=(plots/$st/mu95obs_log/summary_chi2.pdf); done
pdfunite $FILES plots/mu95obs_log_all_studies.pdf
ZFILES=()
for obs in $ZOBS; do ZFILES+=($D/scan_$obs.pdf); done
pdfunite $ZFILES plots/mu95obs_log_zoom200_600_M300_x1.pdf
