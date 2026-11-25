# Signal injection into the hemisphere-mixing pool

Does RPV gluino signal in the data change the mixed-event (hemisphere-mixing)
background estimate? RPV gluino → qqq hemispheres (M300, M1000) are put in the QCD mixing pool
next to the QCD hemispheres. The mixed sample is then rebuilt, and its spectra
are compared with and without signal, as a function of the signal strength μ.
What we learned is in [`results.md`](results.md).

There are four studies: **M300_x1, M300_x4, M1000_x1, M1000_x4**. In x4 the
signal file goes into the pool four times (three copies plus the original), giving
4× the signal hemispheres at the same total cross section.

## Layout

```
signal_injection/
  README.md, results.md
  cmslpc/                      what ran on cmslpc (kept as run)
    run_match_x1.sh, run_match_x4.sh   pairing (run3-mj-match), 2 studies each
    run_gather_seq.sh                  pairs -> EOS, gather jobs, studies 2-4
    config_pairnets.json               evaluator models (copy of mixed_eval/)
    run_inject.sh                      make_histograms.py on the 4 studies + reference
  run_scan.sh                  nominal μ scan: all plots and printed tables
  run_mu95_scan.sh             scan at the CMS 95% CL limit (μ95, 10μ95, 100μ95, μ=1)
  excluded_xs/                 μ95 from CMS-EXO-21-004
    CMS-EXO-21-004_HEPData_Figure4c.csv  (HEPData ins2774247, "Figure 4c")
    mu95_table.py -> mu95_table.txt
  weights/                     how event weights are built (σ/N, product, √)
    weights_table.py -> weights_table.txt
    signal_cutflow.py -> signal_cutflow.txt     (reads EOS: run on cmslpc)
  build_slides.sh              figures + code boxes + all three decks
  slides/
    inject_slides.tex                      walkthrough of the nominal study (13 p.)
    chi2_explainer/chi2_explainer.tex      what Δχ² means here (18 p.), make_figures.py
    mu95_from_cms/mu95_from_cms.tex        μ95 from the CMS limit, dense (6 p.)
    mu95_from_cms/mu95_step_by_step.tex    same, slow, with code and weights (35 p.)
    mu95_from_cms/make_figures.py, code_snippets.py
```

The scan code is one level up, in `../plot_injection_scan.py`. The histogram
maker is `../make_histograms.py`; its `--signal-file-id-min` option splits
events into QQ/SQ/SS.

### Work directory (data, not in git)

Histograms, plots and printouts live in the **work directory**. By default
that is `~/analysis/histograms/inject_26_11_23`; set `INJECT_WORKDIR` to use
another one. Every local script reads it.

```
<work dir>/
  hists_inject_<study>.root, hists_mixed_noninjected.root   from cmslpc/run_inject.sh
  plots/<study>/...,  plots/*_all_studies.pdf               from run_scan.sh, run_mu95_scan.sh
  plots_<study>.txt, plots_mu95obs_<study>.txt              printed tables
  weights/file_tables/file_table_<study>.json               copied from cmslpc (15 MB)
```

`build_slides.sh` links `signal_injection/plots` to `<work dir>/plots`, so the
decks can find the figures. That link and every LaTeX build product are
git-ignored.

## Where everything is

| What | Where |
|---|---|
| Signal input (slimmed, 5-jet) | `/store/group/lpcmultijets/johnny/slimmed_signal_5jet_26_11_12/slimmed_RPV_<M>_ScoutNano_RPV_<M>_1.root` |
| x4 copies | `/store/group/lpcmultijets/johnny/slimmed_signal_5jet_26_11_12_x4copies/..._1_copy{1,2,3}.root` |
| QCD pool | the full-run global index `global_index` (id `bfed2f02`, 5,482 files), 8 slices HT-200to400 and up |
| Mixer work dir | cmslpc `~/nobackup/run3-mj-mixer/inject/` (shards, file tables, indexes, pairs, job dirs), mixer `04b95ad` |
| Mixed events | `/store/user/jlawless/mixing-inject-26-11-12/<study>/{pairs,legs,stitched}/` |
| Evaluated | `/store/group/lpcmultijets/johnny/evaluated_inject_26_11_13/<study>/` (91 files each) |
| Evaluator work dir | cmslpc `~/nobackup/run3-mj-evaluator/inject_eval/`, evaluator `f00ffdf` |
| Non-injected reference | `/store/group/lpcmultijets/johnny/evaluated_mixed_26_11_12/` (same base index, same matching settings) |
| Histograms | cmslpc `~/nobackup/run3-mj-analyzer/inject_26_11_23/`, analyzer `a1d204a`; copied to the local work dir |

In every file table the signal files have `file_id ≥ 5482`: they are added after
the 5,482 QCD files, which keep their ids.

## Re-running the study

Steps 1–5 run on cmslpc, inside `~/nobackup/run3-mj-mixer` (or the evaluator
or analyzer repo), with the LCG_106 view and `pkg-env` active:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
source pkg-env/bin/activate
```

The usual LPC rules apply:
- Every EOS write (`xrdcp`, `xrdfs mkdir`) and every `condor_submit` needs
  John's approval first.
- List EOS directories once and reuse the listing.
- Run `make_histograms.py` with `--inspect-workers 4` on big filesets.

The commands below are the ones that ran, with `$m` = M300 or M1000.

### 1. Put the signal hemispheres in the pool (mixer)

```bash
S=root://cmseos.fnal.gov//store/group/lpcmultijets/johnny/slimmed_signal_5jet_26_11_12
C=root://cmseos.fnal.gov//store/group/lpcmultijets/johnny/slimmed_signal_5jet_26_11_12_x4copies
# x4 only: three byte-identical copies of the signal file (EOS write)
xrdcp -s $S/slimmed_RPV_${m}_ScoutNano_RPV_${m}_1.root $C/slimmed_RPV_${m}_ScoutNano_RPV_${m}_1_copy$i.root

# hemisphere shards of the signal files (and of each copy, for x4)
run3-mj-index $S/slimmed_RPV_${m}_ScoutNano_RPV_${m}_1.root config/config.json --outdir inject/shards

# file table = the full-run table + the signal file(s); w_rel = σ / N
run3-mj-file-table $S/slimmed_RPV_${m}_ScoutNano_RPV_${m}_1.root \
    --extend file_table.json -o inject/file_table_${m}_x1.json --no-progress
run3-mj-file-table $S/slimmed_RPV_${m}_ScoutNano_RPV_${m}_1.root $C/slimmed_RPV_${m}_ScoutNano_RPV_${m}_1_copy{1,2,3}.root \
    --extend file_table.json -o inject/file_table_${m}_x4.json --no-progress

# index = the full-run global index + the signal shard(s)
run3-mj-index-build inject/shards/hindex_slimmed_RPV_${m}_ScoutNano_RPV_${m}_1.root \
    -t inject/file_table_${m}_x1.json --base-index global_index \
    --base-file-table file_table.json -o inject/index_${m}_x1
run3-mj-index-build inject/shards/hindex_slimmed_RPV_${m}_ScoutNano_RPV_${m}_1{,_copy1,_copy2,_copy3}.root \
    -t inject/file_table_${m}_x4.json --base-index global_index \
    --base-file-table file_table.json --max-rows-per-event 4 -o inject/index_${m}_x4
```

### 2. Pair the hemispheres (mixer, login node, about 50 min and 4 GB per study)

[`cmslpc/run_match_x1.sh`](cmslpc/run_match_x1.sh) and
[`cmslpc/run_match_x4.sh`](cmslpc/run_match_x4.sh) run `run3-mj-match` with the
same settings as the non-injected sample: `--max-distance 0.5 --pt-tolerance
0.10 --seed 42 --pairs-per-chunk 100000`. Output goes to `inject/pairs_<study>/`.

### 3. Gather and assemble (mixer, condor)

For each study:
1. Copy `inject/pairs_<study>/pairs_*.root` to `$OUT/pairs`.
2. Generate and submit the gather jobs (60 per study).
3. Check the legs, then generate and submit the assemble jobs.

[`cmslpc/run_gather_seq.sh`](cmslpc/run_gather_seq.sh) did steps 1–2 for three
of the studies in sequence. It contains EOS writes and `condor_submit`, so get
approval before running it.

```bash
OUT=/store/user/jlawless/mixing-inject-26-11-12/$s
W=$PWD/run3_mj_mixer-1.0.0-py3-none-any.whl
python scripts/submitters/submit_gather.py -p root://cmseos.fnal.gov/$OUT/pairs \
    -t inject/file_table_$s.json --manifest inject/pairs_$s/pairs_manifest.json \
    -o $OUT --config config/config.json --wheel $W -P 60 --logdir inject/batch_gather_$s
python scripts/monitors/check_stage_outputs.py legs -p inject/pairs_$s -d /eos/uscms$OUT/legs \
    --gather-manifest inject/batch_gather_$s/gather_manifest.json
python scripts/submitters/submit_assemble.py -p root://cmseos.fnal.gov/$OUT/pairs \
    -l root://cmseos.fnal.gov/$OUT/legs -t inject/file_table_$s.json -o $OUT \
    --config config/config.json --wheel $W --manifest inject/pairs_$s/pairs_manifest.json \
    --gather-manifest inject/batch_gather_$s/gather_manifest.json --logdir inject/batch_assemble_$s
# then: condor_submit inject/batch_{gather,assemble}_$s/submit.sub
```

The assemble jobs write `$OUT/stitched/stitched_00000..00090.root`. Each
event's `xs_weight` is `w_rel[seed] × w_rel[match]`.

### 4. Evaluate (evaluator, condor)

```bash
cd ~/nobackup/run3-mj-evaluator
# fileset: the 91 stitched files of each study (no EOS listing needed)
python3 -c '
import json
for s in ["M300_x1","M300_x4","M1000_x1","M1000_x4"]:
    b=f"root://cmseos.fnal.gov//store/user/jlawless/mixing-inject-26-11-12/{s}/stitched/stitched_"
    json.dump({f"inject_{s}":{"files":{f"{b}{i:05d}.root":"events" for i in range(91)}}},
              open(f"inject_eval/fileset_inject_{s}.json","w"), indent=2)'
python3 scripts/submit_evaluator.py -i inject_eval/fileset_inject_$s.json \
    -o /store/group/lpcmultijets/johnny/evaluated_inject_26_11_13/$s \
    --config inject_eval/config_pairnets.json --wheel run3_mj_evaluator-1.0.0-py3-none-any.whl \
    --logdir inject_eval/batch_$s
```

`config_pairnets.json` (copied here as
[`cmslpc/config_pairnets.json`](cmslpc/config_pairnets.json)) runs four PairNet
models: V0.9, V1, InRepo and GivenByLarry.

### 5. Histograms (analyzer, cmslpc login node, about 10 min for all five)

```bash
bash scripts/signal_injection/cmslpc/run_inject.sh   # -> ~/nobackup/run3-mj-analyzer/inject_26_11_23/
```

The selection: no pSignal cut, mass asymmetry < 0.1, and a fill weight of
`xs_weight^0.5` (the default for mixed events).
- **The four studies:** run with `--signal-file-id-min 5482`, which writes
  QQ/SQ/SS histograms separately.
- **The non-injected August sample:** run without it, as the μ = 0 reference.

Then copy the outputs to the local work dir:

```bash
ssh cmslpc-el9.fnal.gov 'cd ~/nobackup/run3-mj-analyzer/inject_26_11_23 && tar cf - *.root *.log' \
    | tar xf - -C ~/analysis/histograms/inject_26_11_23
```

### 6. Scans and plots (local, a few minutes)

```bash
zsh run_scan.sh          # μ = 0, 1, 3, 10, 100, 1e3, 1e4 (+ short lists), x1 vs x4, vs non-injected
zsh run_mu95_scan.sh     # μ95 table, then μ = 0, μ95, 10μ95, 100μ95, 1; M300 x1 zoom 200-600
```

Both run `../plot_injection_scan.py` in the analyzer's `pkg-env`.
- **x4 studies:** run with `--mu-scale 0.25`, so every μ label is the
  x1-equivalent strength (see `results.md`).
- **Outputs:**
  - Each scan prints a table: signal fraction, Δχ², Z, and the μ at which Δχ²
    reaches 4 and 25. These go to `plots_<study>.txt` and
    `plots_mu95obs_<study>.txt`.
  - The plots go to `plots/<study>/`.

### 7. Weights check and slides (local)

```bash
# file tables: copy once from cmslpc into the work dir
scp 'cmslpc-el9.fnal.gov:nobackup/run3-mj-mixer/inject/file_table_*.json' \
    ~/analysis/histograms/inject_26_11_23/weights/file_tables/
../../pkg-env/bin/python weights/weights_table.py > weights/weights_table.txt
# signal cutflows: reads EOS, so on cmslpc (command in the script's docstring)

zsh build_slides.sh      # every figure, code box and deck; PDFs land next to the .tex
```

The code boxes in `mu95_step_by_step` are not copies. `code_snippets.py`
- finds each line range by searching for text in the real source files (this
  folder, `../plot_injection_scan.py`, `../make_histograms.py`, the mixer, and
  `~/analysis/mc/gen_M1000.log`);
- writes `snippets.tex` with `\lstinputlisting` commands;
- stamps each box with the file's repo and commit.

If a pattern stops matching, the script stops with an error instead of showing
the wrong lines. Rerun it (that is, `build_slides.sh`) after committing, so the
stamps show the new commit.

## Things to keep in mind

- **Weights.**
  - `w_rel = σ / N` per dataset (mixer `filetable.py`). N is the cutflow's
    "All events" bin, taken before any cut and summed over the dataset's files.
    For the signal, N is the count after MLM jet matching (M300 15,254;
    M1000 11,365), normalised to the theory σ.
  - Per event: `xs_weight = w_seed × w_match`, filled with its square root.
  - μ multiplies the signal `w_rel`, so SQ scales as √μ and SS as μ.
- **x4 at μ/4.** The x4 file table already divides `w_rel` by 4. Under √w
  filling, x4 at μ/4 matches x1 at μ in both SQ and SS.
- **Δχ².** The μ and μ = 0 shapes share their QQ events, so Σpull² is a
  noncentrality: read it against 4 and 25 (2σ and 5σ), not against ndf. The
  signal MC's own noise term is subtracted.
- **QCD pool.** Only the HT ≥ 200 slices are in it.
