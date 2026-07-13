#!/usr/bin/env python3
"""event_display.py - 3D EVE event display for run3-mj-mixer output.

Draws, for one event of a ``mixed_*.root`` file:

  * every ScoutingPFJet as a TEveJetCone (dR = 0.4), colored by which side of
    the transverse-thrust plane it falls on (blue = +n_T side, orange = -n_T),
  * the transverse thrust axis n_T as a dashed magenta line,
  * the hemisphere-splitting plane (the plane containing the beam axis,
    perpendicular to n_T) as a translucent gray slab,
  * the two summed Hemisphere four-vectors as thick arrows (lengths scaled to
    the larger |p| of the two),
  * the beam axis as a thin gray line.

Jet-side coloring recomputes the mixer's assignment (p_T projection on n_T
> 0) from ``thrust_axis_phi``, so cones and hemisphere arrows always agree.

Run it INTERACTIVELY so the OpenGL window stays live at the python prompt
(PyROOT's input hook processes GUI events between commands):

    python -i scripts/event_display.py mixed_X.root --event 0

    >>> n()          # next event
    >>> p()          # previous event
    >>> show(42)     # jump to event 42
    >>> save("evt42.png")

Needs ROOT with PyROOT + OpenGL (EVE). On cmslpc use an LCG view; on macOS
`conda install -c conda-forge root` works. Over SSH prefer `ssh -Y`, or copy a
mixed file to your laptop - EVE through forwarded X11 can be sluggish.
"""

import argparse
import math
import sys

import ROOT

R_BARREL = 300.0   # cone/scene extrapolation radius (arbitrary units)
Z_BARREL = 300.0
JET_DR = 0.4       # AK4

COL_POS = ROOT.kAzure + 1    # +n_T side
COL_NEG = ROOT.kOrange + 7   # -n_T side
COL_POS_DARK = ROOT.kAzure - 6
COL_NEG_DARK = ROOT.kOrange + 3


class EventDisplay:
    def __init__(self, path, tree_name="events"):
        self.file = ROOT.TFile.Open(path)
        if not self.file or self.file.IsZombie():
            sys.exit(f"Cannot open {path}")
        self.tree = self.file.Get(tree_name)
        if not self.tree:
            sys.exit(f"No '{tree_name}' tree in {path} (empty mixed file?)")
        self.n_entries = int(self.tree.GetEntries())
        self.index = -1
        self.eve = ROOT.TEveManager.Create()

    # ------------------------------------------------------------------ scene
    def show(self, i):
        if not 0 <= i < self.n_entries:
            print(f"event index {i} out of range [0, {self.n_entries})")
            return
        self.index = i
        t = self.tree
        t.GetEntry(i)
        self.eve.GetEventScene().DestroyElements()

        phi_t = float(t.thrust_axis_phi)
        nx, ny = math.cos(phi_t), math.sin(phi_t)

        self._add_beam_axis()
        self._add_thrust_axis(nx, ny)
        self._add_split_plane(nx, ny)
        n_pos, n_neg = self._add_jets(t, nx, ny)
        self._add_hemisphere_arrows(t)

        self.eve.Redraw3D(ROOT.kTRUE)
        print(
            f"event {i}/{self.n_entries - 1}: thrust={float(t.thrust):.3f} "
            f"phi_T={phi_t:.3f} | split {n_pos}+{n_neg} "
            f"(blue=+n_T, orange=-n_T)"
        )
        for h in range(int(t.nHemisphere)):
            print(
                f"  hemi[{h}] side={int(t.Hemisphere_side[h]):+d} "
                f"njets={int(t.Hemisphere_n_jets[h])} "
                f"pt={float(t.Hemisphere_pt[h]):7.1f} "
                f"mass={float(t.Hemisphere_mass[h]):7.1f} "
                f"eta={float(t.Hemisphere_eta[h]):+6.2f} "
                f"partner_eta={float(t.Hemisphere_partner_eta[h]):+6.2f}"
            )

    def _add_beam_axis(self):
        line = ROOT.TEveLine("beam axis")
        line.SetNextPoint(0.0, 0.0, -Z_BARREL)
        line.SetNextPoint(0.0, 0.0, Z_BARREL)
        line.SetLineColor(ROOT.kGray + 1)
        line.SetLineWidth(1)
        self.eve.AddElement(line)

    def _add_thrust_axis(self, nx, ny):
        line = ROOT.TEveLine("thrust axis n_T")
        line.SetNextPoint(-R_BARREL * nx, -R_BARREL * ny, 0.0)
        line.SetNextPoint(R_BARREL * nx, R_BARREL * ny, 0.0)
        line.SetLineColor(ROOT.kMagenta)
        line.SetLineWidth(3)
        line.SetLineStyle(2)
        self.eve.AddElement(line)

    def _add_split_plane(self, nx, ny):
        # Plane containing the beam axis, perpendicular to n_T: spanned by
        # z-hat and t-hat = (-ny, nx, 0). Drawn as a thin translucent box.
        tx, ty = -ny, nx
        eps = 0.5  # half-thickness of the slab
        box = ROOT.TEveBox("splitting plane")
        # Vertices: two faces offset by +-eps along n_T, each a rectangle in
        # the (t-hat, z) plane.
        verts = []
        for off in (-eps, eps):
            ox, oy = off * nx, off * ny
            verts += [
                (-R_BARREL * tx + ox, -R_BARREL * ty + oy, -Z_BARREL),
                (R_BARREL * tx + ox, R_BARREL * ty + oy, -Z_BARREL),
                (R_BARREL * tx + ox, R_BARREL * ty + oy, Z_BARREL),
                (-R_BARREL * tx + ox, -R_BARREL * ty + oy, Z_BARREL),
            ]
        for j, (x, y, z) in enumerate(verts):
            box.SetVertex(j, x, y, z)
        box.SetMainColor(ROOT.kGray)
        box.SetMainTransparency(85)
        box.SetLineColor(ROOT.kGray + 1)
        self.eve.AddElement(box)

    def _add_jets(self, t, nx, ny):
        n_pos = n_neg = 0
        for j in range(int(t.nScoutingPFJet)):
            pt = float(t.ScoutingPFJet_pt[j])
            eta = float(t.ScoutingPFJet_eta[j])
            phi = float(t.ScoutingPFJet_phi[j])
            proj = pt * (math.cos(phi) * nx + math.sin(phi) * ny)
            pos = proj > 0.0
            n_pos += pos
            n_neg += not pos
            cone = ROOT.TEveJetCone(f"jet {j}: pt={pt:.0f}")
            cone.SetCylinder(R_BARREL, Z_BARREL)
            cone.AddEllipticCone(eta, phi, JET_DR, JET_DR)
            cone.SetMainColor(COL_POS if pos else COL_NEG)
            cone.SetMainTransparency(60)
            cone.SetLineColor(COL_POS if pos else COL_NEG)
            self.eve.AddElement(cone)
        return n_pos, n_neg

    def _add_hemisphere_arrows(self, t):
        n = int(t.nHemisphere)
        mags = [
            math.sqrt(
                float(t.Hemisphere_px[h]) ** 2
                + float(t.Hemisphere_py[h]) ** 2
                + float(t.Hemisphere_pz[h]) ** 2
            )
            for h in range(n)
        ]
        pmax = max(mags) if mags else 1.0
        for h in range(n):
            if mags[h] <= 0.0:
                continue
            scale = 0.85 * R_BARREL * max(mags[h] / pmax, 0.15) / mags[h]
            vx = float(t.Hemisphere_px[h]) * scale
            vy = float(t.Hemisphere_py[h]) * scale
            vz = float(t.Hemisphere_pz[h]) * scale
            arrow = ROOT.TEveArrow(vx, vy, vz, 0.0, 0.0, 0.0)
            side = int(t.Hemisphere_side[h])
            arrow.SetMainColor(COL_POS_DARK if side > 0 else COL_NEG_DARK)
            arrow.SetTubeR(0.015)
            arrow.SetConeR(0.035)
            arrow.SetConeL(0.10)
            arrow.SetElementName(
                f"hemisphere side={side:+d} p={mags[h]:.0f} GeV"
            )
            self.eve.AddElement(arrow)

    # ------------------------------------------------------------- navigation
    def next(self):
        self.show(self.index + 1)

    def prev(self):
        self.show(self.index - 1)

    def save(self, fname):
        self.eve.GetDefaultGLViewer().SavePicture(fname)
        print(f"saved {fname}")


def main():
    parser = argparse.ArgumentParser(
        description="3D EVE event display for mixed_*.root files. Run with "
        "`python -i` and navigate with n()/p()/show(i)/save('x.png')."
    )
    parser.add_argument("input", help="mixed ROOT file")
    parser.add_argument("--tree", default="events")
    parser.add_argument("--event", type=int, default=0, help="first event to show")
    args = parser.parse_args()

    display = EventDisplay(args.input, args.tree)
    display.show(args.event)
    return display


if __name__ == "__main__":
    _display = main()
    # Interactive helpers for `python -i`:
    show = _display.show
    n = _display.next
    p = _display.prev
    save = _display.save
    if not hasattr(sys, "ps1") and not sys.flags.interactive:
        print(
            "\nTip: run with `python -i scripts/event_display.py <file>` to "
            "keep the viewer live and navigate with n()/p()/show(i)."
        )
        ROOT.gApplication.Run()
