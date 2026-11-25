# Signal injection: results

**Question.** Hemisphere mixing builds the QCD background from the data
themselves. If the data contain RPV gluino signal, some of the hemispheres in
the mixing pool are signal. Does that change the mixed background?

**Answer.** Not at any signal strength that is still allowed. At the CMS 95% CL
excluded cross section, the mixed spectra change by Δχ² ≤ 0.3 in every
observable; 2σ needs Δχ² = 4. M300 would need 15–105× its allowed cross
section to make a 2σ change, and M1000 170–730×. At the full theory cross section
(μ = 1, already excluded) M300 would be clearly visible in HT and lead-jet pT.

How to reproduce all of this is in [`README.md`](README.md).
- Plots: `~/analysis/histograms/inject_26_11_23/plots/`.
- Decks: `slides/` (built with `build_slides.sh`).

## Setup

- **Signal:** RPV gluino → qqq at M300 and M1000, 13.6 TeV, normalised to the
  theory σ (637.6 pb and 0.4842 pb).
  - The signal hemispheres go into the full-run QCD pool (8 HT slices,
    200to400 and up; 5,482 files), next to the QCD hemispheres.
  - Matching settings are as in the non-injected sample: pT ±10%, max distance
    0.5, seed 42.
- **Four studies:** each mass at x1 (the signal file once) and x4 (the file
  four times, each copy at ¼ weight). x4 has 4× the signal hemispheres, which
  tests whether the result depends on signal MC statistics.
- **Pair types:** every mixed event is QQ, SQ or SS, depending on how many of
  its two hemispheres are signal.
  - The fill weight is √(xs_weight), so at signal strength μ the content is
    B(μ) = QQ + √μ·SQ + μ·SS.
  - x4 studies are run at μ/4, which gives the same SQ and SS totals as x1 at
    μ.
- **Selection:** mass asymmetry < 0.1, no pSignal cut.
- **Observables:** HT, lead-jet pT, and the paired trijet mass from four
  PairNet models (V0.9, V1, InRepo, GivenByLarry).
- **Measure:** the unit-area shape at μ is compared with μ = 0 bin by bin.
  - Because both shapes share their QQ events, Σpull² is a Δχ² (a
    noncentrality). It reads as √Δχ² σ for a known-shape test, so 4 means 2σ
    and 25 means 5σ.
  - The SQ sample's own MC-noise term is subtracted.
  - The precision is that of the ~8M-event mixed MC, not of data.

Entries after cuts, per study:

| | QQ | SQ | SS |
|---|---|---|---|
| M300 x1 | 8,066,315 | 1,468 | 0 |
| M300 x4 | 8,064,235 | 5,751 | 7 |
| M1000 x1 | 8,066,487 | 480 | 0 |
| M1000 x4 | 8,066,579 | 1,908 | 2 |

SS is negligible, as expected. All the signal effect comes from SQ.

## Checks

1. **Injection does not disturb the QCD pairing.** Each study's μ = 0 (QQ only)
   was compared with the non-injected mixed sample. That sample has the same
   base index and settings, but its pairings differ because the extra signal
   rows change the random draws.
   - Yields agree to within 0.6% (ratios 0.9948–1.0009).
   - χ²/ndf ≤ 1 in every observable of every study (largest 47.7/48).
2. **x1 and x4 agree.**
   - x4 has 3.9–4.0× the SQ entries of x1, as it should.
   - At the same μ, HT and lead-jet pT give the same Δχ² to within 10% at
     both masses. For example, HT reaches 2σ at μ = 0.066 in both M300
     studies.
   - The mass models agree within ~30% at M300. At M1000 they differ by up
     to 2×: x4 reaches 2σ at lower μ. M1000 x1 has only 480 SQ entries, so
     its mass shapes carry real MC noise; x4 is the more reliable of the two.
   - Neither difference changes any conclusion below, which holds with
     margins of 15× or more.
3. **The weights are right** (`weights/`, step-by-step deck).
   - `w_rel = σ/N`, where N is the "All events" count before any cut. For
     the signal that is the count after MLM matching: M300 15,254, M1000
     11,365.
   - x4 divides `w_rel` by 4.
   - Mixed events carry `w_seed × w_match` and are filled with its square
     root.
   - μ multiplies the signal `w_rel`.

## Nominal-strength scan (μ in units of the theory cross section)

Δχ² at μ = 1, x1 studies:

| | HT | lead-jet pT | mass (4 PairNets) |
|---|---|---|---|
| M300 | 60.4 | 77.6 | 12.0–20.7 |
| M1000 | 0.8 | 0.2 | 0.2–0.5 |

Signal strength needed for a given Δχ² (x4 in brackets):

| | HT | lead-jet pT | mass (4 PairNets) |
|---|---|---|---|
| M300, 2σ (Δχ² = 4) | 0.066 (0.067) | 0.052 (0.053) | 0.19–0.33 (0.21–0.36) |
| M300, 5σ (Δχ² = 25) | 0.41 (0.42) | 0.32 (0.33) | 1.2–2.1 (1.3–2.2) |
| M1000, 2σ | 5.0 (5.3) | 20 (22) | 8.2–17 (5.3–10) |
| M1000, 5σ | 31 (33) | 124 (138) | 52–105 (32–61) |

- **M300:** most of the effect is in the kinematics, not in mass. Signal
  hemispheres paired with QCD hemispheres shift HT and lead-jet pT more than
  they build a mass peak. Of the mass models, InRepo responds most and V0.9
  least.
- **M1000:** the signal is ~10⁻⁵ of the mixed weight at μ = 1 and changes
  nothing visible.

## At the excluded cross section (μ95)

The latest CMS limit is CMS-EXO-21-004 (PRL 133 (2024) 201803, 128 fb⁻¹,
13 TeV, scouting); it excludes gluinos up to 1.7 TeV. Its HEPData table
"Figure 4c" gives observed and expected limits on σ·B·A, and the theory σ·A.
- **The ratio:** μ95 = limit / theory σ·A, so the acceptance cancels.
- **Check:** μ95 crosses 1 between 1.6 and 1.8 TeV, which reproduces the
  paper's 1.7 TeV exclusion.
- **Our energy:** the same fraction is applied to our 13.6 TeV theory σ.

| | μ95 observed | μ95 expected (±1σ) | allowed σ at 13.6 TeV |
|---|---|---|---|
| M300 | 3.39×10⁻³ | 1.99 [1.06, 3.61]×10⁻³ | 2.16 pb |
| M1000 | 0.0300 | 0.038 [0.024, 0.064] | 0.0145 pb |

The observed limit is used: it is what the data allow. At M300 it is weaker
than expected, just inside +1σ.

Largest Δχ² over all observables (x1; x4 is within 0.1 at μ95 and 10 μ95):

| | μ95 | 10 μ95 | 100 μ95 | μ = 1 |
|---|---|---|---|---|
| M300 | 0.3 | 2.7 (lead pT) | 26 (lead pT); mass 4.1–7.1 | 78 |
| M1000 | 0.0 | 0.2 | 2.4 (HT) | 0.8 |

At μ95 the signal is 3–6×10⁻⁴ of the mixed weight for M300, and ~10⁻⁶ for
M1000.

Multiples of μ95 needed for 2σ:

| | HT | lead-jet pT | mass |
|---|---|---|---|
| M300 | 19× | 15× | 57–105× |
| M1000 | 170× | 660–730× | 175–560× |

## Conclusions

1. **At allowed signal strengths, signal in the mixing pool does not change the
   mixed background.** Every observable stays within Δχ² ≤ 0.3 of the
   signal-free shape at the observed 95% CL limit. That holds for both masses
   and for x1 and x4.
2. **The margin is large.**
   - M300 would have to be 15–20× above its limit to show up at 2σ in HT or
     lead-jet pT, and 57–105× in mass.
   - M1000 would have to be more than 170× above its limit.
3. **The method notices signal when there is enough of it.** At μ = 1, M300 is
   a 7.8σ (HT) and 8.8σ (lead-jet pT) change. So the null result at μ95 is a
   real statement, not an insensitive test.
4. **The mixing itself is not disturbed.** μ = 0 matches the non-injected
   sample in yield and shape. x1 and x4 agree in HT and lead-jet pT, so the
   result does not hinge on signal MC statistics. The M1000 mass curves are
   the noisiest part, but they are hundreds of times μ95 away from mattering.

## Caveats

- **Precision.** The Δχ² uses the mixed-MC statistics (~8M QQ entries,
  luminosity 1, unit-area shapes). It answers "would the shape change at all",
  not "how big would the bias be in the final fit". A data-size estimate
  would scale Δχ² by the ratio of effective statistics.
- **Shape only.** Normalisation is not tested: the shapes are compared at unit
  area. The signal fraction of the total weight, printed in
  `plots_*.txt`, bounds any normalisation effect: 3–6×10⁻⁴ at M300 μ95.
- **The μ95 translation assumes** the signal scales from 13 to 13.6 TeV like
  the theory cross section (ratios 1.38 and 1.49), and that the 2016–18
  scouting limit applies to our selection as an excluded fraction of the
  theory σ.
- **Generic test.** The significance quoted is for a known-shape (template)
  test. A generic shape test, which does not know where the signal is, is
  weaker. The `Z_gen` column of the printouts gives it; at μ95 it is 0.0
  everywhere.
- **Mass points.** Only two were studied. Between them the conclusion should
  hold, since μ95 rises with mass while the signal's share of the pool falls.
