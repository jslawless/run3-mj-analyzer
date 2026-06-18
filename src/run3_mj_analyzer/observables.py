"""observables.py - simple kinematic observables for candidate tri-jets."""

import numpy as np

__all__ = ["mass_asymmetry"]


def mass_asymmetry(m1, m2):
    """Mass asymmetry between an event's two candidate tri-jets: |m1 - m2| / |m1 + m2|.

    ``m1`` / ``m2`` are the two tri-jet masses; they may be scalars or matching
    numpy / awkward arrays (the expression is plain elementwise arithmetic and
    broadcasts). Bounded to [0, 1] for physical (non-negative) masses: 0 when the
    two tri-jets have equal mass, 1 when one is massless.
    """
    return np.abs(m1 - m2) / np.abs(m1 + m2)
