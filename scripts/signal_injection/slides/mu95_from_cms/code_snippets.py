"""Write snippets.tex: \\lstinputlisting commands that show line ranges of the
real source files, so the deck displays the files themselves, never a copy.

Each range is found by text, not hard-coded line numbers, so it follows the
code if the file changes. A begin/end pattern that is missing or matches more
than once is an error. Run before every compile:

    python code_snippets.py && latexmk -pdf mu95_step_by_step.tex
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INJ = HERE.parents[1]                      # .../scripts/signal_injection
WORK = Path(os.environ.get("INJECT_WORKDIR",     # plots and printouts
                           "~/analysis/histograms/inject_26_11_23")).expanduser()
ANALYZER = INJ.parents[1]
SCAN = ANALYZER / "scripts/plot_injection_scan.py"
MKHIST = ANALYZER / "scripts/make_histograms.py"
MIXER = Path.home() / "analysis/run3-mj-mixer/src/run3_mj_mixer"
WEIGHTS = INJ / "weights"
GENLOG = Path.home() / "analysis/mc/gen_M1000.log"

#: key: (file, [(begin text, end text), ...]); each range runs from the line
#: containing ``begin`` to the first line at or after it containing ``end``.
#: Matching is on the line with surrounding whitespace stripped, by prefix;
#: a pattern starting with "=" must equal the whole stripped line.
SNIPPETS = {
    "limitdiv": (INJ / "excluded_xs/mu95_table.py",
                 [("def mu95_table(", "return t")]),
    "limitout": (INJ / "excluded_xs/mu95_table.txt",
                [("M region", "M region"), ("300    low", "300    low"),
                 ("1000   high", "1000   high")]),
    "combine": (SCAN, [("def combine(", "return v, var")]),
    "muargs": (SCAN, [("if args.mu95:", "mus = sorted({0.0, 1.0")]),
    "dchisq": (SCAN, [("for i, mu in enumerate(nonzero):",
                      "v, var = combine(parts, mu * mu_scale"),
                     ("frac = (v.sum() - v0.sum())", "raw, noise)")]),
    "readn": (MIXER / "filetable.py",
              [("def read_cutflow0(path):", 'return float(f["cutflow"].values()[0])')]),
    "wrel": (MIXER / "filetable.py",
             [("for ds, n0 in sorted(n_original.items()):",
               '"w_rel": float(xs_pb) / float(n0),')]),
    "xsweight": (MIXER / "assemble.py",
                 [("w = ft.w_rel_by_id(table)",
                   'xs_weight = w[pairs["seed_file_id"]]')]),
    "fillweight": (MKHIST,
                   [("# The file's own weight, raised to --weight-power",
                     "np.float64) ** args.weight_power")]),
    "pairsplit": (MKHIST, [("n_sig = sum(", "w if np.isscalar(w) else w[sel])")]),
    "wtable": (WEIGHTS / "weights_table.txt",
               [("Per-dataset weight", "RPV_M1000 (x4)")]),
    "wpairs": (WEIGHTS / "weights_table.txt",
               [("Example mixed events", "SQ at mu = 1")]),
    "sigcutflow": (WEIGHTS / "signal_cutflow.txt",
                   [("RPV_M300 cutflow", "HT > 400 GeV"),
                    ("RPV_M1000 cutflow", "HT > 400 GeV")]),
    "genlog": (GENLOG, [("Before matching:", "Filter efficiency (event-level)")]),
    "runloop": (INJ / "run_mu95_scan.sh",
                [("typeset -A MU95", "typeset -A MU95"),
                 ("=for st in $STUDIES; do", "done")]),
    "scanout": (WORK / "plots_mu95obs_M300_x1.txt",
                [("observable                       mu", "observable                       mu"),
                 ("ht  ", "ht  "),
                 ("mass_PairNetV0_9 ", "mass_PairNetV0_9 ")]),
}


def match(line, pattern):
    if pattern.startswith("="):
        return line == pattern[1:]
    return line.startswith(pattern)


def find_ranges(path, pairs):
    lines = [l.strip() for l in path.read_text().splitlines()]
    out = []
    for begin, end in pairs:
        hits = [i for i, l in enumerate(lines) if match(l, begin)]
        if begin == end:            # the first run of consecutive matches
            if not hits:
                sys.exit(f"{path.name}: no line matches '{begin}'")
            last = hits[0]
            while last + 1 in hits:
                last += 1
            out.append((hits[0] + 1, last + 1))
            continue
        if len(hits) != 1:
            sys.exit(f"{path.name}: '{begin}' matches {len(hits)} lines")
        b = hits[0]
        e = next((i for i in range(b, len(lines))
                  if match(lines[i], end)), None)
        if e is None:
            sys.exit(f"{path.name}: no '{end}' after line {b + 1}")
        out.append((b + 1, e + 1))
    return out


def provenance(path):
    """(path to show, 'repo@commit' (+ 'uncommitted changes')). In git the
    path is relative to the repo; otherwise it is the full path, 'not in git'."""
    d = path.parent
    home = str(path).replace(str(Path.home()), "~")
    try:
        top = subprocess.run(["git", "-C", d, "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return home, "not in git"
    top = Path(top.stdout.strip())
    repo = top.name
    commit = subprocess.run(["git", "-C", d, "log", "-1", "--format=%h"],
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", d, "status", "--porcelain", "--",
                            path.name], capture_output=True, text=True).stdout
    return (str(path.resolve().relative_to(top.resolve())),
            f"{repo}@{commit}" + (", with uncommitted changes" if dirty else ""))


def tex_escape(s):
    return s.replace("_", r"\_").replace("~", r"\textasciitilde{}")


#: listings reads a file line by line up to the last line shown, which is
#: minutes for a 50k-line log. Above this size the shown lines are written out
#: to excerpts/<key>.txt at every build (same text, numbered as in the source).
EXCERPT_ABOVE = 2000


def excerpt(key, path, ranges):
    """(file to include, linerange string) for a big file's ranges."""
    src = path.read_text().splitlines()
    (HERE / "excerpts").mkdir(exist_ok=True)
    lines, spans, n = [], [], 0
    for a, b in ranges:
        lines += src[a - 1:b]
        spans.append((n + 1, n + b - a + 1, a))
        n += b - a + 1
    out = HERE / "excerpts" / f"{key}.txt"
    out.write_text("\n".join(lines) + "\n")
    return out, spans


def main():
    out = ["% generated by code_snippets.py - do not edit"]
    for key, (path, pairs) in SNIPPETS.items():
        ranges = find_ranges(path, pairs)
        rel = os.path.relpath(path, HERE)
        linerange = ",".join(f"{a}-{b}" for a, b in ranges)
        first = ""
        if len(path.read_text().splitlines()) > EXCERPT_ABOVE:
            if len(ranges) != 1:
                sys.exit(f"{key}: excerpts support one range only")
            ex, spans = excerpt(key, path, ranges)
            rel = os.path.relpath(ex, HERE)
            linerange = f"{spans[0][0]}-{spans[0][1]}"
            first = f"firstnumber={ranges[0][0]},"
        shown = ", ".join(f"{a}--{b}" if a != b else f"{a}" for a, b in ranges)
        shown_path, prov = provenance(path)
        lang = {".py": "language=Python,", ".sh": "language=bash,"}.get(
            path.suffix, "style=plainout,")
        out.append(
            f"\\newcommand{{\\snip{key}}}{{\\lstinputlisting"
            f"[{lang}{first}linerange={{{linerange}}},consecutivenumbers=false]"
            f"{{{rel}}}}}")
        out.append(
            f"\\newcommand{{\\src{key}}}{{\\texttt{{{tex_escape(shown_path)}}}, "
            f"lines {shown} ({tex_escape(prov)})}}")
        print(f"{key:<10} {shown_path}: {linerange}")
    (HERE / "snippets.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
