"""CAD-derived nominal collision pads and K3 jaw-command calibration."""

import numpy as np

# CTAG2F90-00 STEP pads bound 23 mm x 39.5 mm x 6 mm. K3's
# grasp planner defines the closing direction as the tool's local X.
PAD_THICKNESS_M = 0.006
PAD_WIDTH_M = 0.023
PAD_LENGTH_M = 0.0395
PAD_MASS_KG = 0.065
PAD_FRICTION = 1.15
OPENING_POINTS = ((0.2, 0.021), (0.3, 0.031), (0.5, 0.050), (1.0, 0.088))
MAX_OPENING_M = OPENING_POINTS[-1][1]
# The CTAG2F90 drawing gives 185 mm overall depth open and 202.5 mm
# closed. Fit a circular pad path with a parallel jaw orientation from the
# drawing's closed and open endpoints.
AXIAL_TRAVEL_M = 0.0175
LINK_RADIUS_M = ((MAX_OPENING_M / 2) ** 2 + AXIAL_TRAVEL_M ** 2) / (2 * AXIAL_TRAVEL_M)
DRIVE_STIFFNESS = 5000.0
DRIVE_DAMPING = 20.0
DRIVE_FORCE_N = 11.5


def opening_m(command):
    """Interpolate measured command ratios. Extrapolate towards zero opening."""
    if not 0 <= command <= 1:
        raise ValueError("Gripper ratio must be in [0, 1].")
    ratios, widths = zip((0.0, 0.0), *OPENING_POINTS)
    return float(np.interp(command, ratios, widths))


def axial_retraction_m(half_gap):
    half_gap = np.clip(half_gap, 0.0, MAX_OPENING_M / 2)
    return float(LINK_RADIUS_M - np.sqrt(LINK_RADIUS_M ** 2 - half_gap ** 2))
