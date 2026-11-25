"""The cutflow of each injected signal file: bin 0 ("All events") is the N in
w_rel = xs_pb / N. Reads EOS, so run it on cmslpc (paths from the file table):

    ssh cmslpc-el9.fnal.gov 'cd ~/nobackup/run3-mj-mixer && \
        X509_USER_PROXY=~/x509up_u59599 pkg-env/bin/python -' \
        < signal_cutflow.py > signal_cutflow.txt
"""
import uproot

BASE = ("root://cmseos.fnal.gov//store/group/lpcmultijets/johnny/"
        "slimmed_signal_5jet_26_11_12/slimmed_RPV_{m}_ScoutNano_RPV_{m}_1.root")

for m in ("M300", "M1000"):
    with uproot.open(BASE.format(m=m)) as f:
        h = f["cutflow"]
        print(f"RPV_{m} cutflow")
        for label, n in zip(h.axis().labels(), h.values()):
            print(f"  {label:<32} {n:>8,.0f}")
