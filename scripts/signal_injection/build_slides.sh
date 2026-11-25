#!/bin/zsh
# Rebuild the three slide decks from the study's plots and histograms in the
# work directory (default ~/analysis/histograms/inject_26_11_23, or
# $INJECT_WORKDIR). Links <this dir>/plots -> <work dir>/plots, which the
# inject and chi2 decks read through \graphicspath.
set -e
HERE=${0:A:h}
ANALYZER=${HERE:h:h}
PY=${PY:-$ANALYZER/pkg-env/bin/python}
export INJECT_WORKDIR=${INJECT_WORKDIR:-~/analysis/histograms/inject_26_11_23}
ln -sfn $INJECT_WORKDIR/plots $HERE/plots
LTX=(latexmk -pdf -interaction=nonstopmode -quiet)

cd $HERE/slides
$LTX inject_slides.tex

cd $HERE/slides/chi2_explainer
$PY make_figures.py
$LTX chi2_explainer.tex

cd $HERE/slides/mu95_from_cms
$PY make_figures.py
$PY code_snippets.py           # snippets.tex: the code boxes, read from the real files
$LTX mu95_from_cms.tex
$LTX mu95_step_by_step.tex
