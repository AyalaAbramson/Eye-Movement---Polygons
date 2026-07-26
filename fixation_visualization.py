"""
Visualize eye-tracking fixations overlaid on target polygon stimuli.

All trials are identical across participants, so the same trial can be compared
across sessions. The module exposes three visualization modes:

    - plot_single_subject(session_folder, target_trial)   one subject, one trial
    - plot_aggregated_subjects(root_dir, target_trial)    all subjects, one trial
    - plot_multi_trials(root_dir, ...)                    several trials, side by side

Fixations can be drawn as a scatter or as a density heatmap, and the multi-trial
mode can lay out either all rotations of one polygon, or all variations (steps)
of one vertex-count at a fixed rotation.

Time synchronization between the EyeLink clock (ASC file) and the experiment
clock (CSV) is performed using the first ``TRIAL_START`` sync message.

>>> Edit the CONFIGURATION block below to choose what to plot, then run the file.
"""

import os
import glob
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerPatch
from scipy.stats import gaussian_kde, ttest_ind, ttest_rel, ttest_1samp, linregress
from scipy.stats import t as t_dist
from shapely.geometry import Polygon, Point


class _HandlerEllipse(HandlerPatch):
    """Draw Ellipse legend keys as a little ellipse instead of the default box,
    so the Gaussian-fit entry clearly matches the ellipse drawn in the graph."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        center = (width / 2.0 - xdescent, height / 2.0 - ydescent)
        e = Ellipse(xy=center, width=width + xdescent, height=height + ydescent)
        e.set_facecolor(orig_handle.get_facecolor())
        e.set_edgecolor(orig_handle.get_edgecolor())
        e.set_linewidth(orig_handle.get_linewidth())
        e.set_linestyle(orig_handle.get_linestyle())
        e.set_transform(trans)
        return [e]


# Apply to every legend in the process (Ellipse patches -> ellipse glyphs).
Legend.update_default_handler_map({Ellipse: _HandlerEllipse()})


# =====================================================================
# CONFIGURATION  (edit here, then run the file -- see main() at bottom)
# =====================================================================

# --- Paths ---
ROOT_DIR = r'C:\Users\User\Desktop\דברים\לימודים\שנה ג\מחקר מודרך\Analysis\data\raw'
# One example session, used for single-subject mode and for reading the trial
# list in multi-trial mode (trials are identical across participants).
EXAMPLE_SESSION = os.path.join(
    ROOT_DIR, 'participant_315328062', 'session_20260611_153617')

# --- Stimulus geometry (must match run_experiment.py / core_functions.py) ---
SCREEN_W, SCREEN_H = 2560, 1440   # screen resolution
IMAGE_SIZE = 1100                 # stimulus image space (run_experiment IMAGE_SIZE)
BASE_RADIUS = 150                 # base polygon radius (core_functions)

# --- Distance units (for the spatial-distance statistics graphs) ---
# 'px'  : report offsets / axis lengths in pixels.
# 'deg' : convert them to visual-angle degrees (psychophysical convention),
#         using the viewing geometry below. Affects only the distance graphs
#         (offsets, axis lengths) and their axis labels -- not the scatter maps.
DISTANCE_UNITS = 'deg'        # 'px' | 'deg'
VIEWING_DISTANCE_CM = 60.0   # eye-to-screen distance
SCREEN_WIDTH_CM = 52.0       # physical width of the display (it is SCREEN_W px wide)

# --- View / zoom ---
# 'polygon' zooms each panel to the stimulus (+ its fixations) so the polygon is
# large and easy to analyse; the on-screen position is no longer preserved.
# 'screen' keeps the true 0..SCREEN_W x 0..SCREEN_H layout.
VIEW = 'polygon'          # 'polygon' | 'screen'
VIEW_MARGIN_PX = 120      # padding (px) around the polygon/fixations when zoomed

# --- What to plot ---
PLOT_MODE = 'analytics'      # 'single' | 'aggregated' | 'multi' | 'analytics'
TARGET_TRIAL = 4          # used by 'single' and 'aggregated'

# --- Category colors (uniform across every graph) ---
FILL_COLORS = {True: '#3498DB', False: '#E67E22'}   # filled = blue, unfilled = orange

# --- All-data analytics (PLOT_MODE == 'analytics') ---
# Analytical graphs ONLY (no polygon/fixation panels), pooled over EVERY trial
# (all sides / rotations / steps):
#   1. ellipse axis lengths, filled vs unfilled
#   2. ellipse-center offset from original centroid & center of mass
#   3. mean ellipse-center offset vs step size
# ANALYTICS_FILL controls whether graphs 2 & 3 split by fill ('both') or show a
# single condition ('filled' | 'unfilled'); graph 1 always compares both.
ANALYTICS_FILL = 'both'   # 'filled' | 'unfilled' | 'both'
SHOW_CENTER_BIAS = True     # add the screen-center bias analysis figure
SHOW_RAW_POINTS = True      # overlay individual-trial points on the mean/SD plots
# Restrict the analytics dataset to specific polygon sides / rotations / stretch
# steps (this does NOT filter by participant). None = use all values.
ANALYTICS_SIDES = None     # e.g. 7 or [5, 6] to keep only those vertex counts
ANALYTICS_ROTATIONS = None  # e.g. [0, 90] to keep only those rotations
ANALYTICS_STEPS = None      # e.g. [3, 4] to keep only those stretch steps
# When True, fit a per-participant ellipse per trial and AVERAGE within each
# subject+condition first, so every plotted point / statistical observation is a
# subject mean (repeated-measures), not an individual trial. When False, all
# participants' fixations are pooled into one ellipse per trial (the default).
USE_SUBJECT_AVERAGES = False

# --- Statistical significance annotations (analytics graphs) ---
# Annotate statistically significant points/comparisons with an asterisk (*).
#   - Bar charts: Welch t-test comparing Filled vs Unfilled within each category.
#   - Line graphs: Welch t-test comparing each point against the baseline (the
#     first x value) of the same curve.
# When several curves share an axes, the asterisk takes its curve's color (and
# is vertically staggered) for readability; a single curve uses black.
SHOW_SIGNIFICANCE = True   # toggle the significance asterisks
SIGNIFICANCE_ALPHA = 0.05  # p-value threshold for marking a point significant

# --- Fixation time window ---
# Saccadic latency: subjects keep fixating the central cross for ~150-250 ms
# after the polygon appears. Discard fixations that start within this window
# after stimulus onset so that carry-over central fixation is not counted.
# Set to 0.0 to keep every in-window fixation.
STIM_ONSET_LATENCY_S = 0.3   # seconds after stim_on to ignore

# --- Spatial filter: distance from the polygon boundary ---
# Keep only fixations close to the polygon. None disables the filter; otherwise
# the value is a distance in pixels, interpreted per FIXATION_BOUNDARY_MODE:
#   'outside' : drop fixations more than this many px OUTSIDE the polygon
#               (interior fixations are always kept) -- removes far-away strays.
#   'band'    : keep only fixations within this distance of the boundary LINE on
#               either side (drops deep-interior and far-exterior fixations).
FIXATION_BOUNDARY_DIST_PX = 100     # e.g. 80; None = no spatial filtering
FIXATION_BOUNDARY_MODE = 'outside'   # 'outside' | 'band'

# --- Fixation rendering ---
FIXATION_STYLE = 'scatter'   # 'scatter' | 'heatmap'
HEATMAP_CMAP = 'inferno'    # any matplotlib colormap name
HEATMAP_GRID = 240          # density grid resolution per axis (higher = smoother)
HEATMAP_BW = 0.35           # KDE bandwidth scale; larger = smoother/blobbier
HEATMAP_ALPHA = 0.8         # opacity of the density layer
HEATMAP_THRESH = 0.08       # hide density below this fraction of the peak

# --- Centroid markers ---
SHOW_ACTUAL_CENTROID = True   # geometric center of the drawn polygon (yellow)
SHOW_BASE_CENTROID = True     # intended stimulus center from the CSV (black)

# --- Bivariate Gaussian (2D normal) modeling ---
# Fit a 2D Gaussian to the fixation cloud and overlay a confidence ellipse.
SHOW_GAUSSIAN_ELLIPSE = True  # overlay the fitted confidence ellipse
ELLIPSE_N_STD = 1.8           # ellipse radius in standard deviations (~95% at 2)
ELLIPSE_ONLY = False          # draw ONLY the ellipse + polygon (hide fixations);
                              # the title then lists the fitted parameters
SHOW_STATS_PLOT = False       # in 'multi' mode, also emit the statistical-inference plot

# --- Multi-trial layout (PLOT_MODE == 'multi') ---
# 'rotations' : fix (sides, step), show every rotation of that polygon.
# 'variations': fix (sides, rotation), show every step/variation at that rotation.
MULTI_GROUP_BY = 'variations'
MULTI_SIDES =  7     # number of polygon vertices to select
MULTI_STEP =  None            # used when MULTI_GROUP_BY == 'rotations'
MULTI_ROTATION = None    # used when MULTI_GROUP_BY == 'variations'
MULTI_AGGREGATE = True    # True: pool fixations across all sessions; False: EXAMPLE_SESSION only
# Each geometric polygon was shown twice: once image-filled, once not. Choose
# which to display so the figure is not overcrowded.
MULTI_FILL = 'filled'     # 'filled' | 'unfilled' | 'both'

# --- Experiment trial-generation parameters (mirror run_experiment.py) ---
# The CSV does not store the image-fill flag, but the trial order is fully
# determined by these values plus the seeded shuffle, so fill is recoverable.
# Keep these in sync with run_experiment.py if that file changes.
EXP_SEED = 42
EXP_POLYGON_TYPES = [5, 6, 7]
EXP_STEPS = [-1, 0, 1, 3, 4]      # 0 = flat edge; 1 = no stretch (single rotation)
EXP_ROTATIONS = [0, 45, 90, 135, 180]
EXP_FILL_OPTIONS = [True, False]
EXP_TRIAL_REPETITIONS = 1

# =====================================================================
# (end configuration)
# =====================================================================


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def parse_polygon_str(poly_str):
    """Parse a 'x,y;x,y;...' polygon string into a list of (x, y) tuples."""
    if pd.isna(poly_str) or not poly_str:
        return []
    points = []
    for pair in str(poly_str).split(';'):
        if ',' in pair:
            x, y = map(float, pair.split(','))
            points.append((x, y))
    return points


def _resolve_asc_path(session_folder):
    """Return the path to the single .asc file inside the session's edf folder."""
    edf_dir = os.path.join(session_folder, 'edf')
    asc_files = glob.glob(os.path.join(edf_dir, '*.asc'))
    if not asc_files:
        raise FileNotFoundError(f"No .asc file found in {edf_dir}")
    return asc_files[0]


def _read_time_offset(asc_file_path):
    """
    Compute the EyeLink->experiment clock offset from the first TRIAL_START msg.

    Message format:  MSG <eyelink_ms> TRIAL_START <exp_clock_s> index=N type=...

        time_offset = (eyelink_ms / 1000.0) - exp_clock_s
    """
    with open(asc_file_path, 'r') as f:
        for line in f:
            if 'TRIAL_START' not in line:
                continue
            parts = line.split()
            # parts: ['MSG', '<eyelink_ms>', 'TRIAL_START', '<exp_clock_s>', ...]
            try:
                eyelink_ms = float(parts[1])
                exp_clock_s = float(parts[3])
            except (IndexError, ValueError):
                continue
            return (eyelink_ms / 1000.0) - exp_clock_s
    raise ValueError(f"No TRIAL_START message found in {asc_file_path}")


def _place_polygon(polygon, offset):
    """
    Map polygon vertices from the image space onto the screen by translation
    only (Offset Mapping) -- no scaling/stretching.

    The polygon is authored in image space (IMAGE_SIZE in run_experiment.py) and
    the stimulus image is blitted at its top-left corner on screen, so adding
    that offset lands the polygon in the same screen-pixel coordinate system as
    the fixations while preserving its native size and aspect ratio.
    """
    offset_x, offset_y = offset
    return [(x + offset_x, y + offset_y) for x, y in polygon]


def _filter_by_boundary_distance(xs, ys, polygon, dist, mode):
    """
    Keep only fixations within `dist` pixels of the polygon, per `mode`.

        'outside' : distance measured to the polygon AREA (0 inside) -- keeps
                    interior fixations and any within `dist` px outside the edge.
        'band'    : distance measured to the boundary LINE -- keeps a band of
                    width `dist` on either side of the edge (drops deep interior
                    and far exterior).

    Returns the filtered (xs, ys). No-ops when `dist` is None or the polygon has
    fewer than 3 vertices.
    """
    if dist is None or not polygon or len(polygon) < 3:
        return xs, ys

    poly = Polygon(polygon)
    if not poly.is_valid:
        poly = poly.buffer(0)          # repair minor self-intersections
    ref = poly.exterior if mode == 'band' else poly

    kept_x, kept_y = [], []
    for x, y in zip(xs, ys):
        if ref.distance(Point(x, y)) <= dist:
            kept_x.append(x)
            kept_y.append(y)
    return kept_x, kept_y


def _load_trial_row(session_folder, target_trial):
    """Return the CSV row (Series) for `target_trial`, or None if absent."""
    trials_csv_path = os.path.join(session_folder, 'trials.csv')
    df = pd.read_csv(trials_csv_path)
    df = df[df['trial_index'].notna()].copy()
    df['trial_index'] = df['trial_index'].astype(int)

    trial_data = df[df['trial_index'] == target_trial]
    if trial_data.empty:
        return None
    return trial_data.iloc[0]


# --------------------------------------------------------------------------- #
# Trial classification (recover sides / rotation / step from polygon geometry)
# --------------------------------------------------------------------------- #
def classify_trial(trial_row):
    """
    Recover the generative parameters of an auto polygon from its geometry.

    The CSV does not store the vertex count, rotation, or stretch step directly,
    but they are fully determined by the points (see generate_auto_polygon):
        - sides    = number of vertices.
        - rotation = angle of base vertex 0 about the image center (deg).
        - step     = stretch applied to vertex 0 (target vertex). -1 = collapsed.

    Returns a dict {'sides', 'rotation', 'step'} or None if there is no polygon.
    """
    base = parse_polygon_str(trial_row.get('polygon_base_points_imgspace'))
    actual = parse_polygon_str(trial_row.get('polygon_points_imgspace'))
    if not base or len(base) < 3:
        return None

    sides = len(base)
    cx = cy = IMAGE_SIZE / 2.0

    # Rotation: vertex 0 sits at angle (-pi/2 + rotation_rad) about the center.
    bx0, by0 = base[0]
    rotation_rad = math.atan2(by0 - cy, bx0 - cx) + (math.pi / 2.0)
    rotation = round(math.degrees(rotation_rad)) % 360

    # Step: radius of the (stretched) target vertex 0 relative to the flat edge.
    step = None
    if actual:
        r0 = math.hypot(actual[0][0] - cx, actual[0][1] - cy)
        flat_radius = BASE_RADIUS * math.cos(2 * math.pi / sides)
        delta = BASE_RADIUS - flat_radius
        if r0 < 1.0:
            step = -1                       # fully collapsed vertex
        elif delta != 0:
            step = round((r0 - flat_radius) / delta)

    return {'sides': sides, 'rotation': rotation, 'step': step}


def trial_param_table(session_folder):
    """Return a list of {'trial_index', 'sides', 'rotation', 'step'} for a session."""
    trials_csv_path = os.path.join(session_folder, 'trials.csv')
    df = pd.read_csv(trials_csv_path)
    df = df[df['trial_index'].notna()].copy()
    df['trial_index'] = df['trial_index'].astype(int)

    table = []
    for _i, row in df.iterrows():
        params = classify_trial(row)
        if params is None:
            continue
        params['trial_index'] = int(row['trial_index'])
        table.append(params)
    return table


def build_fill_map():
    """
    Reconstruct {trial_index -> {'sides','step','rotation','is_filled'}}.

    The CSV stores no image-fill flag, but the experiment assigns trial_index by
    1-based position in the auto-combos list (see run_full_experiment), and that
    list is produced by the deterministic, seeded shuffle below. Replaying it
    therefore recovers each trial's fill state. Geometry (sides/step/rotation) is
    included so callers can validate the mapping against the actual CSV points.
    """
    import random

    def get_rotations(step):
        # step == 1 has no stretch, so all rotations look identical -> use one.
        return [EXP_ROTATIONS[0]] if step == 1 else EXP_ROTATIONS

    base_combos = [
        (sides, step, rot, is_filled)
        for sides in EXP_POLYGON_TYPES
        for step in EXP_STEPS
        for rot in get_rotations(step)
        for is_filled in EXP_FILL_OPTIONS
    ]
    combos = base_combos * EXP_TRIAL_REPETITIONS
    random.seed(EXP_SEED)
    random.shuffle(combos)

    return {
        i + 1: {'sides': s, 'step': st, 'rotation': rot % 360, 'is_filled': fill}
        for i, (s, st, rot, fill) in enumerate(combos)
    }


def select_trials(session_folder, group_by, sides, step=None, rotation=None,
                  fill='both'):
    """
    Select trial indices for a multi-trial figure.

    group_by == 'rotations' : keep trials with the given `sides` and `step`,
                              return them ordered by rotation (all rotations of
                              the same polygon).
    group_by == 'variations': keep trials with the given `sides` and `rotation`,
                              return them ordered by step (all variations of the
                              same vertex-count at one rotation).

    fill == 'filled' | 'unfilled' | 'both' restricts to image-filled polygons,
    plain (black) polygons, or both. The fill flag is recovered from the seeded
    trial order (build_fill_map) and validated against each trial's geometry.

    Returns a list of param dicts (each with 'trial_index' and 'is_filled')
    ordered by the varying parameter.
    """
    # Accept a single value or a list/tuple/set for step / rotation.
    def _as_set(v, mod=None):
        if v is None:
            return None
        seq = v if isinstance(v, (list, tuple, set)) else [v]
        return {(x % mod if mod is not None else x) for x in seq}

    step_set = _as_set(step)
    rot_set = _as_set(rotation, mod=360)

    table = trial_param_table(session_folder)
    fill_map = build_fill_map()
    mismatches = 0
    matches = []
    for t in table:
        if t['sides'] != sides:
            continue
        if group_by == 'rotations' and step_set is not None and t['step'] not in step_set:
            continue
        if group_by == 'variations' and rot_set is not None and t['rotation'] not in rot_set:
            continue

        # Recover the fill flag, but only trust it when the reconstructed
        # geometry matches the CSV geometry for this trial_index.
        combo = fill_map.get(t['trial_index'])
        is_filled = None
        if (combo and combo['sides'] == t['sides']
                and combo['step'] == t['step']
                and combo['rotation'] == t['rotation']):
            is_filled = combo['is_filled']
        elif combo is not None:
            mismatches += 1

        t = dict(t, is_filled=is_filled)
        if fill == 'filled' and is_filled is not True:
            continue
        if fill == 'unfilled' and is_filled is not False:
            continue
        matches.append(t)

    if mismatches and fill != 'both':
        print(f"Warning: fill mapping disagreed with geometry on {mismatches} "
              f"trial(s); EXP_* config may be out of sync with run_experiment.py.")

    # Order by the varying parameter first, then the other one (so multi-valued
    # selections lay out tidily).
    primary = 'rotation' if group_by == 'rotations' else 'step'
    secondary = 'step' if group_by == 'rotations' else 'rotation'
    matches.sort(key=lambda t: (t[primary] is None, t[primary],
                                t[secondary] is None, t[secondary]))
    return matches


# --------------------------------------------------------------------------- #
# Core extraction
# --------------------------------------------------------------------------- #
def extract_valid_fixations(session_folder, target_trial,
                            boundary_dist=FIXATION_BOUNDARY_DIST_PX,
                            boundary_mode=FIXATION_BOUNDARY_MODE):
    """
    Extract time-synced, in-window fixations for `target_trial` in a session.

    Steps:
        1. Read stim window (stim_on_ts_s, stim_off_ts_s) and polygon from CSV.
        2. Compute the clock offset from the first TRIAL_START message.
        3. Parse EFIX lines, align each fixation start to the CSV clock:
               aligned_start_s = (fixation_start_ms / 1000.0) - time_offset
        4. Keep only fixations whose aligned_start_s is within the stim window,
           starting STIM_ONSET_LATENCY_S after onset (to drop the carry-over
           central cross fixation caused by saccadic latency).
        5. Optionally keep only fixations within `boundary_dist` px of the
           polygon boundary (see _filter_by_boundary_distance / the
           FIXATION_BOUNDARY_* config).

    Returns a dict with keys:
        'x', 'y'            -> lists of fixation coordinates (screen pixels)
        'polygon'           -> list of (x, y) vertices (offset-mapped to screen)
        'base_centroid'     -> (stim_center_x_px, stim_center_y_px) from the CSV
        'stim_on', 'stim_off'
        'time_offset'
    or None if the trial is not present in the CSV.
    """
    trial_row = _load_trial_row(session_folder, target_trial)
    if trial_row is None:
        return None

    stim_on = float(trial_row['stim_on_ts_s'])
    stim_off = float(trial_row['stim_off_ts_s'])
    polygon = parse_polygon_str(trial_row['polygon_points_imgspace'])

    # Offset Mapping: translate the polygon from image space to its on-screen
    # position (stimulus top-left) so it aligns with the fixations, undistorted.
    stim_offset = (float(trial_row['stim_top_left_x_px']),
                   float(trial_row['stim_top_left_y_px']))
    polygon = _place_polygon(polygon, stim_offset)

    # Original centroid: the intended stimulus center recorded in the CSV. Already
    # in screen pixels, so no offset mapping is needed.
    base_centroid = (float(trial_row['stim_center_x_px']),
                     float(trial_row['stim_center_y_px']))

    asc_file_path = _resolve_asc_path(session_folder)
    time_offset = _read_time_offset(asc_file_path)

    fixations_x = []
    fixations_y = []

    with open(asc_file_path, 'r') as f:
        for line in f:
            if not line.startswith('EFIX'):
                continue
            parts = line.split()
            # EFIX R  start_ms  end_ms  dur_ms  x_px  y_px  pupil
            if len(parts) < 7:
                continue
            try:
                fixation_start_ms = float(parts[2])
                x = float(parts[5])
                y = float(parts[6])
            except ValueError:
                # Skip rows with missing/corrupt samples (e.g. '.')
                continue

            aligned_start_s = (fixation_start_ms / 1000.0) - time_offset
            # Skip the saccade-latency window after onset so the carry-over
            # central (cross) fixation is not counted.
            window_start = stim_on + STIM_ONSET_LATENCY_S
            if window_start <= aligned_start_s <= stim_off:
                fixations_x.append(x)
                fixations_y.append(y)

    # Optional spatial filter: keep only fixations near the polygon boundary.
    fixations_x, fixations_y = _filter_by_boundary_distance(
        fixations_x, fixations_y, polygon, boundary_dist, boundary_mode)

    return {
        'x': fixations_x,
        'y': fixations_y,
        'polygon': polygon,
        'base_centroid': base_centroid,
        'stim_on': stim_on,
        'stim_off': stim_off,
        'time_offset': time_offset,
    }


def find_session_folders(root_dir):
    """
    Return every session folder under `root_dir`.

    A session folder is any directory that directly contains a 'trials.csv'.
    This handles the root/participant_*/session_*/ layout as well as sessions
    placed directly under the root.
    """
    sessions = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        if 'trials.csv' in filenames:
            sessions.append(dirpath)
    return sorted(sessions)


def collect_fixations(target_trial, sessions):
    """
    Pool valid fixations for `target_trial` across one or more sessions.

    Returns (x, y, polygon, base_centroid, n_sessions). The polygon and base
    centroid come from the first session that provides them (identical across
    participants).
    """
    all_x, all_y = [], []
    polygon = []
    base_centroid = None
    n_sessions = 0

    for session_folder in sessions:
        data = extract_valid_fixations(session_folder, target_trial)
        if data is None:
            continue
        n_sessions += 1
        all_x.extend(data['x'])
        all_y.extend(data['y'])
        if not polygon and data['polygon']:
            polygon = data['polygon']
        if base_centroid is None:
            base_centroid = data['base_centroid']

    return all_x, all_y, polygon, base_centroid, n_sessions


# --------------------------------------------------------------------------- #
# Statistics (pure math -- no plotting)
# --------------------------------------------------------------------------- #
def fit_gaussian(x, y, n_std=ELLIPSE_N_STD, ref_point=None):
    """
    Fit a bivariate normal distribution to the (x, y) fixation coordinates.

    Uses numpy.mean for the center and numpy.cov for the covariance, then derives
    the confidence-ellipse geometry from the eigen-decomposition of the
    covariance matrix (eigenvectors give the axis directions, eigenvalues their
    variances).

    Orientation in a full 0..360 framework
    --------------------------------------
    An eigenvector's sign is arbitrary, so the bare major-axis angle is only
    defined modulo 180 deg -- which makes line graphs of orientation vs polygon
    rotation 'wrap' and break beyond 180 deg. To recover a true 0..360 reading we
    anchor the major-axis direction to the displacement vector from `ref_point`
    (the stimulus original centroid) to the fixation mean: the major eigenvector is
    flipped to point into the same half-plane as that displacement, then atan2
    yields a sign-preserving 0..360 angle.

    Returns a clean dict of computed parameters, or None when there are too few
    or degenerate points to form a valid (positive-definite) covariance:

        {
          'mean'        : (mx, my),          # distribution center
          'cov'         : 2x2 numpy array,
          'var_x','var_y','cov_xy',          # covariance entries
          'angle_deg'     : major-axis orientation, [0, 180)  (textbook; ellipse)
          'angle_deg_360' : direction-resolved orientation, [0, 360)
          'offset_angle_deg' : direction of (mean - ref_point), [0, 360) or None
          'offset_dist'   : distance from ref_point to mean (px) or None
          'semi_major','semi_minor',         # n_std * sqrt(eigenvalue)
          'width','height',                  # full axis lengths (for Ellipse)
          'n_std', 'n'                       # confidence level and sample size
        }
    """
    if x is None or len(x) < 3:
        return None

    pts = np.vstack([x, y])
    # Need a non-degenerate (rank-2) cloud, otherwise the covariance is singular.
    centered = pts - pts.mean(axis=1, keepdims=True)
    if np.linalg.matrix_rank(centered) < 2:
        return None

    mean = pts.mean(axis=1)
    cov = np.cov(pts)                       # 2x2
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending, eigvecs columns
    if np.any(eigvals <= 0):
        return None

    # Order largest-eigenvalue (major axis) first.
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major_vec = eigvecs[:, 0]
    # Textbook principal angle (mod 180), used to draw the (symmetric) ellipse.
    angle_deg = math.degrees(math.atan2(major_vec[1], major_vec[0])) % 180.0

    # Resolve the major-axis direction to a full 0..360 using the base->mean
    # displacement as the directional anchor.
    offset_angle = None
    offset_dist = None
    resolved_vec = major_vec
    if ref_point is not None:
        dx = mean[0] - ref_point[0]
        dy = mean[1] - ref_point[1]
        offset_dist = math.hypot(dx, dy)
        if offset_dist > 1e-9:
            offset_angle = math.degrees(math.atan2(dy, dx)) % 360.0
            # Flip the eigenvector into the displacement's half-plane.
            if (major_vec[0] * dx + major_vec[1] * dy) < 0:
                resolved_vec = -major_vec
    angle_deg_360 = math.degrees(
        math.atan2(resolved_vec[1], resolved_vec[0])) % 360.0

    semi_major = n_std * math.sqrt(eigvals[0])
    semi_minor = n_std * math.sqrt(eigvals[1])

    return {
        'mean': (float(mean[0]), float(mean[1])),
        'cov': cov,
        'var_x': float(cov[0, 0]),
        'var_y': float(cov[1, 1]),
        'cov_xy': float(cov[0, 1]),
        'angle_deg': float(angle_deg),
        'angle_deg_360': float(angle_deg_360),
        'offset_angle_deg': (float(offset_angle) if offset_angle is not None
                             else None),
        'offset_dist': (float(offset_dist) if offset_dist is not None
                        else None),
        'semi_major': float(semi_major),
        'semi_minor': float(semi_minor),
        'width': float(2.0 * semi_major),
        'height': float(2.0 * semi_minor),
        'n_std': float(n_std),
        'n': int(pts.shape[1]),
    }


def format_stats(stats):
    """One-line summary of the fitted Gaussian for use in a plot title."""
    mx, my = stats['mean']
    return (f"μ=({mx:.0f}, {my:.0f})   "
            f"σ²x={stats['var_x']:.0f}  σ²y={stats['var_y']:.0f}  "
            f"cov={stats['cov_xy']:.0f}   θ={stats['angle_deg_360']:.1f}°(360)   "
            f"axes={stats['semi_major']:.0f}/{stats['semi_minor']:.0f}px")


def _euclidean(a, b):
    """Euclidean distance between two (x, y) points; None if either is missing."""
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def px_to_visual_deg(px):
    """Convert a length in screen pixels to visual angle (degrees).

    Uses the standard size-to-angle formula 2*atan((L/2)/d) with the viewing
    geometry from the CONFIGURATION block (SCREEN_WIDTH_CM gives the pixel
    pitch, VIEWING_DISTANCE_CM the eye-to-screen distance).
    """
    cm_per_px = SCREEN_WIDTH_CM / SCREEN_W
    half_cm = (px * cm_per_px) / 2.0
    return 2.0 * math.degrees(math.atan2(half_cm, VIEWING_DISTANCE_CM))


def to_distance(px):
    """Convert a pixel distance to the configured DISTANCE_UNITS ('px'|'deg')."""
    if px is None:
        return None
    return px_to_visual_deg(px) if DISTANCE_UNITS == 'deg' else px


def dist_unit_label():
    """Axis-label unit string for the configured distance units."""
    return 'deg' if DISTANCE_UNITS == 'deg' else 'px'


def welch_p(sample_a, sample_b):
    """Welch's two-sample t-test p-value, or None when a sample is too small."""
    if sample_a is None or sample_b is None:
        return None
    if len(sample_a) < 2 or len(sample_b) < 2:
        return None
    try:
        return float(ttest_ind(sample_a, sample_b, equal_var=False).pvalue)
    except Exception:
        return None


def paired_p(sample_a, sample_b):
    """Paired t-test p-value (same trials, two measures); None if unusable."""
    if sample_a is None or sample_b is None:
        return None
    if len(sample_a) != len(sample_b) or len(sample_a) < 2:
        return None
    try:
        return float(ttest_rel(sample_a, sample_b).pvalue)
    except Exception:
        return None


def is_significant(p, alpha=None):
    """True when p is a valid p-value below the significance threshold."""
    if alpha is None:
        alpha = SIGNIFICANCE_ALPHA
    return p is not None and p == p and p < alpha   # p == p rejects NaN


def _fmt_p(p):
    """Format a p-value for console printing (with a significance marker)."""
    if p is None or p != p:                 # None or NaN
        return 'n/a'
    star = ' *' if p < SIGNIFICANCE_ALPHA else ''
    return (f'{p:.2e}{star}' if p < 0.001 else f'{p:.3f}{star}')


def _annotate_star(ax, x, y, color='black'):
    """Place a significance asterisk just above (x, y) in the given color."""
    ax.annotate('*', xy=(x, y), textcoords='offset points', xytext=(0, 1),
                ha='center', va='bottom', color=color, fontsize=15,
                fontweight='bold', clip_on=True)


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def _square_bounds(cx, cy, half, margin=0.0):
    """Return a square view (xmin, xmax, ymin, ymax) centered on (cx, cy)."""
    half = max(half + margin, 1.0)
    return (cx - half, cx + half, cy - half, cy + half)


def _actual_centroid(polygon):
    """Geometric centroid (x, y) of the polygon, or None if it is degenerate."""
    if polygon and len(polygon) >= 3:
        c = Polygon(polygon).centroid
        return (c.x, c.y)
    return None


def _image_bounds(polygon, base_centroid):
    """
    Default square view for a SINGLE plot: an IMAGE_SIZE-wide window centered on
    the stimulus (its original centroid), i.e. the on-screen footprint of the
    stimulus image. Used when there is nothing else in the set to compare to.
    """
    if base_centroid is not None:
        cx, cy = base_centroid
    elif polygon:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    else:
        cx, cy = SCREEN_W / 2.0, SCREEN_H / 2.0
    return _square_bounds(cx, cy, IMAGE_SIZE / 2.0)


def _draw_polygon(ax, polygon):
    """Draw polygon outline + light fill, and return its centroid (x, y)."""
    if not polygon:
        return None

    px, py = zip(*polygon)
    px_closed = list(px) + [px[0]]
    py_closed = list(py) + [py[0]]
    # Boundary at a high zorder so it stays visible over a heatmap.
    ax.plot(px_closed, py_closed, color='#2C3E50', linewidth=2.5,
            label='Polygon boundary', zorder=6)
    ax.fill(px_closed, py_closed, color='#3498DB', alpha=0.15,
            label='Polygon interior', zorder=1)

    if len(polygon) >= 3:
        centroid = Polygon(polygon).centroid
        return (centroid.x, centroid.y)
    return None


def _draw_fixations(ax, x, y, style, color, alpha, label, bounds=None):
    """
    Draw fixations either as a scatter or a smooth density heatmap.

    The heatmap uses a Gaussian kernel density estimate (KDE) evaluated on a
    fine grid, so it renders as an organic 'blob' that follows the fixation
    distribution instead of the blocky bins a 2D histogram produces.

    `bounds` (xmin, xmax, ymin, ymax) restricts the KDE grid to the view region
    so the density stays sharp when zoomed in; None uses the full screen.

    Returns the heatmap mappable (for an optional colorbar), or None.
    """
    if not x:
        return None

    if style == 'heatmap':
        return _draw_kde_heatmap(ax, x, y, bounds)

    ax.scatter(x, y, color=color, s=60, alpha=alpha,
               edgecolors='white', linewidths=0.4, label=label, zorder=5)
    return None


def _draw_kde_heatmap(ax, x, y, bounds=None):
    """
    Render a smooth Gaussian-KDE density of the fixations.

    The density is evaluated on a grid spanning `bounds` (or the full screen when
    None). Falls back to a scatter when there are too few distinct points to
    estimate a 2D density (KDE needs a non-degenerate point cloud).
    """
    pts = np.vstack([x, y])
    # gaussian_kde needs at least 3 points that are not all collinear/identical;
    # guard against the singular-covariance error on tiny samples.
    if pts.shape[1] < 3 or np.linalg.matrix_rank(pts - pts.mean(axis=1, keepdims=True)) < 2:
        ax.scatter(x, y, color='#E74C3C', s=60, alpha=0.85,
                   edgecolors='white', linewidths=0.4, label='Fixations', zorder=5)
        return None

    kde = gaussian_kde(pts, bw_method=HEATMAP_BW)

    x0, x1, y_top, y_bottom = bounds if bounds else (0, SCREEN_W, 0, SCREEN_H)
    gx = np.linspace(x0, x1, HEATMAP_GRID)
    gy = np.linspace(y_top, y_bottom, HEATMAP_GRID)
    mesh_x, mesh_y = np.meshgrid(gx, gy)
    density = kde(np.vstack([mesh_x.ravel(), mesh_y.ravel()])).reshape(mesh_x.shape)

    # Mask away near-zero density so the blob blends into the background instead
    # of tinting the whole panel.
    peak = density.max()
    if peak > 0:
        density = np.ma.masked_less(density, HEATMAP_THRESH * peak)

    # imshow with origin='upper' matches the inverted (top-left origin) Y axis.
    mappable = ax.imshow(
        density, extent=[x0, x1, y_bottom, y_top], origin='upper',
        cmap=HEATMAP_CMAP, alpha=HEATMAP_ALPHA, aspect='auto',
        interpolation='bilinear', zorder=2)
    return mappable


def _draw_gaussian_ellipse(ax, stats, color='#00E5FF'):
    """Overlay the fitted confidence ellipse and its center onto `ax`."""
    if stats is None:
        return
    ellipse = Ellipse(
        xy=stats['mean'], width=stats['width'], height=stats['height'],
        angle=stats['angle_deg'], facecolor='none', edgecolor=color,
        linewidth=2.2, linestyle='-', zorder=9,
        label=f"Gaussian {stats['n_std']:.0f}σ")
    ax.add_patch(ellipse)
    # Mark the distribution mean (ellipse center) on top of everything else
    # (above the centroid markers and fixations).
    ax.scatter([stats['mean'][0]], [stats['mean'][1]], color=color, s=40,
               marker='x', linewidths=2.0, zorder=20)


def _draw_centroid(ax, actual_centroid, base_centroid,
                   show_actual=True, show_base=True):
    """Plot the polygon centroid(s) on top of everything else.

    actual_centroid : geometric center of the drawn polygon (yellow dot).
    base_centroid   : intended stimulus center from the CSV (black dot).
    Use show_actual / show_base to toggle which markers are drawn. Both use
    the same marker size.
    """
    size = 20
    if show_actual and actual_centroid is not None:
        ax.scatter([actual_centroid[0]], [actual_centroid[1]], color='#F1C40F',
                   s=size, marker='o', edgecolors='black', linewidths=2.0,
                   label='Center of mass', zorder=11)
    if show_base and base_centroid is not None:
        ax.scatter([base_centroid[0]], [base_centroid[1]], color='black',
                   s=size, marker='o', edgecolors='white', linewidths=1.0,
                   label='Original centroid', zorder=12)


def _format_axes(ax, title, legend=True, bounds=None, title_fontsize=12):
    """
    Apply shared formatting (inverted Y axis). When `bounds` is given the view
    is zoomed to it; otherwise the full screen (0..SCREEN_W x 0..SCREEN_H).
    """
    ax.set_title(title, fontsize=title_fontsize, pad=6)
    ax.set_xlabel('X coordinate (pixels)', fontsize=9)
    ax.set_ylabel('Y coordinate (pixels)', fontsize=9)
    ax.tick_params(labelsize=8)
    if bounds:
        x0, x1, y_top, y_bottom = bounds
        ax.set_xlim(x0, x1)
        ax.set_ylim(y_bottom, y_top)   # invert Y: smaller y at the top
    else:
        ax.set_xlim(0, SCREEN_W)
        ax.set_ylim(SCREEN_H, 0)       # invert Y: (0,0) at the top-left
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.6)
    if legend:
        ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=8)


def _draw_trial_data(ax, x, y, polygon, base_centroid, title, fixation_style,
                     color, alpha, show_actual, show_base, bounds,
                     legend=True, title_fontsize=12,
                     show_ellipse=SHOW_GAUSSIAN_ELLIPSE,
                     ellipse_only=ELLIPSE_ONLY):
    """Draw one trial's polygon + fixations + centroids using explicit `bounds`.

    `bounds` (xmin, xmax, ymin, ymax) is applied verbatim, so callers can share
    one global window across a set of panels. Aspect is forced to 'equal' in
    _format_axes, keeping polygons geometrically proportioned at any scale.

    When `show_ellipse` (or `ellipse_only`) is set, a 2D Gaussian is fitted to
    the fixations and its confidence ellipse is overlaid. `ellipse_only` hides
    the raw fixations/heatmap and appends the fitted parameters to the title.

    Returns (n_fixations, heatmap_mappable_or_None, gaussian_stats_or_None).
    """
    actual_centroid = _draw_polygon(ax, polygon)

    mappable = None
    if not ellipse_only:
        mappable = _draw_fixations(ax, x, y, fixation_style, color, alpha,
                                   label='Fixations', bounds=bounds)

    stats = None
    if show_ellipse or ellipse_only:
        stats = fit_gaussian(x, y, ref_point=base_centroid)
        if stats is not None:
            _draw_gaussian_ellipse(ax, stats)
        else:
            print(f"Warning: too few/degenerate fixations to fit a Gaussian "
                  f"ellipse ({len(x) if x else 0} point(s)); skipping ellipse.")

    _draw_centroid(ax, actual_centroid, base_centroid,
                   show_actual=show_actual, show_base=show_base)

    if ellipse_only and stats is not None:
        title = f"{title}\n{format_stats(stats)}"
    _format_axes(ax, title, legend=legend, bounds=bounds,
                 title_fontsize=title_fontsize)
    return (len(x) if x else 0), mappable, stats


def _render_trial(ax, target_trial, sessions, title, fixation_style,
                  color, alpha, show_actual, show_base, legend=True,
                  title_fontsize=12, bounds=None,
                  show_ellipse=SHOW_GAUSSIAN_ELLIPSE, ellipse_only=ELLIPSE_ONLY):
    """Collect and draw a single trial onto `ax`.

    When `bounds` is None and VIEW == 'polygon', the view defaults to the
    IMAGE_SIZE-wide stimulus window (a single plot has nothing to compare
    against). Pass explicit `bounds` to share one window across a set.

    Returns (n_fixations, heatmap_mappable_or_None, gaussian_stats_or_None).
    """
    x, y, polygon, base_centroid, _n = collect_fixations(target_trial, sessions)

    if bounds is None and VIEW == 'polygon':
        bounds = _image_bounds(polygon, base_centroid)

    return _draw_trial_data(ax, x, y, polygon, base_centroid, title,
                            fixation_style, color, alpha, show_actual, show_base,
                            bounds, legend=legend, title_fontsize=title_fontsize,
                            show_ellipse=show_ellipse, ellipse_only=ellipse_only)


# --------------------------------------------------------------------------- #
# Mode A: single subject
# --------------------------------------------------------------------------- #
def plot_single_subject(session_folder, target_trial,
                        fixation_style=FIXATION_STYLE,
                        show_actual=SHOW_ACTUAL_CENTROID,
                        show_base=SHOW_BASE_CENTROID,
                        show_ellipse=SHOW_GAUSSIAN_ELLIPSE,
                        ellipse_only=ELLIPSE_ONLY):
    """Plot the polygon + valid fixations for one participant / one trial."""
    fig, ax = plt.subplots(figsize=(9, 9))
    n_fix, mappable, _stats = _render_trial(
        ax, target_trial, [session_folder],
        title=f"Fixations - Trial {target_trial}\n"
              f"{os.path.basename(session_folder)}",
        fixation_style=fixation_style, color='#2ECC71', alpha=0.85,
        show_actual=show_actual, show_base=show_base,
        show_ellipse=show_ellipse, ellipse_only=ellipse_only)

    print(f"[single] {os.path.basename(session_folder)}: {n_fix} valid fixations")
    if mappable is not None:
        fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04,
                     label='Fixation density')
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Mode B: aggregated subjects
# --------------------------------------------------------------------------- #
def plot_aggregated_subjects(root_dir, target_trial,
                             fixation_style=FIXATION_STYLE,
                             show_actual=SHOW_ACTUAL_CENTROID,
                             show_base=SHOW_BASE_CENTROID,
                             show_ellipse=SHOW_GAUSSIAN_ELLIPSE,
                             ellipse_only=ELLIPSE_ONLY):
    """
    Plot valid fixations for `target_trial` across ALL sessions under root_dir.

    Low scatter alpha (or the heatmap style) reveals fixation density where
    points overlap.
    """
    sessions = find_session_folders(root_dir)
    if not sessions:
        print(f"No session folders found under {root_dir}")
        return

    fig, ax = plt.subplots(figsize=(9, 9))
    n_fix, mappable, _stats = _render_trial(
        ax, target_trial, sessions,
        title=f"Aggregated fixations - Trial {target_trial}\n"
              f"{len(sessions)} sessions",
        fixation_style=fixation_style, color='#E74C3C', alpha=0.25,
        show_actual=show_actual, show_base=show_base,
        show_ellipse=show_ellipse, ellipse_only=ellipse_only)

    print(f"[agg] {n_fix} fixations from {len(sessions)} sessions "
          f"for trial {target_trial}")
    if mappable is not None:
        fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04,
                     label='Fixation density')
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Automated statistical inference for the multi-trial dashboard
# --------------------------------------------------------------------------- #
def _fill_word(is_filled):
    return {True: 'Filled', False: 'Unfilled'}.get(is_filled, 'Unknown')


def plot_multi_statistics(stats_records, group_by, fill):
    """
    Emit the statistical-inference figure(s) that correspond to the grouped
    independent variable in the multi-trial dashboard:

      - group_by == 'rotations'  -> ellipse orientation angle vs polygon rotation.
      - group_by == 'variations' -> distance(original centroid, ellipse center) vs step.
      - fill == 'both'           -> additionally, a boxplot contrasting the
                                    semi-major / semi-minor axis lengths between
                                    the filled and unfilled conditions.

    Records whose Gaussian fit failed (stats is None) are skipped with a note.
    """
    usable = [r for r in stats_records if r['stats'] is not None]
    skipped = len(stats_records) - len(usable)
    if skipped:
        print(f"[stats] skipped {skipped} trial(s) without a valid Gaussian fit.")
    if not usable:
        print("[stats] no valid Gaussian fits; skipping statistics plot.")
        return

    fills_present = sorted({r['is_filled'] for r in usable},
                           key=lambda v: v is not True)

    # Tag stating which trial set these statistics were computed from.
    src = 'rotations data' if group_by == 'rotations' else 'steps data'

    # ---- Main plot driven by the grouped variable -------------------------- #
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    cyclic_360 = (group_by == 'rotations')
    if group_by == 'rotations':
        xlabel = 'Polygon rotation (deg)'
        ylabel = 'Model orientation θ (deg, unwrapped)'
        title = f'Model orientation vs polygon rotation ({src})'
        xkey = 'rotation'
        # Use the direction-resolved 0..360 angle so rotations past 180 deg do
        # not wrap and break the linear trend.
        yfunc = lambda r: r['stats']['angle_deg_360']
    else:  # 'variations'
        xlabel = 'Stretch step level'
        ylabel = f'Distance: original centroid → ellipse center ({dist_unit_label()})'
        title = f'Centroid offset vs stretch step ({src})'
        xkey = 'step'
        yfunc = lambda r: to_distance(
            _euclidean(r['base_centroid'], r['stats']['mean']))

    unwrapped_y = []   # collect plotted y-values to set a dynamic ylim later
    for is_filled in fills_present:
        series = sorted((r for r in usable if r['is_filled'] == is_filled),
                        key=lambda r: r[xkey])
        xs = [r[xkey] for r in series]
        ys = [yfunc(r) for r in series]
        if cyclic_360 and len(ys) > 1:
            # Phase-unwrap the cyclical orientation so the underlying linear
            # trend is continuous (no artificial jumps at the 0/360 seam).
            # np.unwrap works in radians; convert in and back out of degrees.
            ys = list(np.degrees(np.unwrap(np.radians(ys), period=2 * np.pi)))
        unwrapped_y.extend(v for v in ys if v is not None)
        if len(fills_present) > 1:
            label = _fill_word(is_filled)
        elif group_by == 'rotations':
            label = 'Orientation θ'   # named so it joins the combined legend
        else:
            label = None
        # Single trend line gets a fixed color per metric (orientation vs
        # centroid offset); multiple conditions keep distinct cycle colors.
        trend_color = 'cornflowerblue' if group_by == 'rotations' else 'darkslateblue'
        line_color = trend_color if len(fills_present) == 1 else None
        ax.plot(xs, ys, 'o-', color=line_color, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.6)
    if cyclic_360 and unwrapped_y:
        # Fit the Y-axis to the unwrapped (continuous) range so the full upward
        # trend is shown across panels without being clipped.
        lo, hi = min(unwrapped_y), max(unwrapped_y)
        pad = max(10.0, 0.08 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        # Keep the continuous (unwrapped) line, but relabel the ticks back into
        # standard 0..360 circle notation via modulo 360 (positions unchanged).
        ticks = ax.get_yticks()
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t % 360:.0f}°" for t in ticks])
        ax.set_ylim(lo - pad, hi + pad)   # re-assert (set_yticks can rescale)

    # Legend in a row beneath the plot (so it never hides the data).
    if len(fills_present) > 1:
        handles, lbls = ax.get_legend_handles_labels()
        fig.legend(handles, lbls, loc='outside lower center',
                   ncol=len(handles), fontsize=8)

    # ---- rotations: ellipse-center offsets in a SEPARATE figure ------------- #
    # The offsets are in pixels (not degrees), so they get their own plot rather
    # than sharing the orientation axis.
    if group_by == 'rotations':
        figo, axo = plt.subplots(figsize=(7, 5), constrained_layout=True)
        # Color encodes the fill condition (matching the axis-length bar chart:
        # Filled = blue, Unfilled = orange); linestyle encodes which centroid.
        # All points use the SAME marker.
        offset_metrics = [
            ('Offset from original centroid', 'base_centroid', '-'),
            ('Offset from center of mass', 'actual_centroid', '--'),
        ]
        legend_handles, legend_labels = [], []
        for is_filled in fills_present:
            series = sorted((r for r in usable if r['is_filled'] == is_filled),
                            key=lambda r: r['rotation'])
            xs = [r['rotation'] for r in series]
            cond = f" ({_fill_word(is_filled)})" if len(fills_present) > 1 else ""
            fcolor = FILL_COLORS.get(is_filled, 'gray')
            for name, key, ls in offset_metrics:
                ds = [to_distance(_euclidean(r[key], r['stats']['mean']))
                      for r in series]
                axo.plot(xs, ds, linestyle=ls, marker='o', color=fcolor)
                # Line-only legend handle (no marker) so the color (fill) and
                # solid/dashed (centroid) are unmistakable.
                legend_handles.append(
                    Line2D([0], [0], color=fcolor, linestyle=ls, linewidth=2))
                legend_labels.append(f"{name}{cond}")
        axo.set_xlabel('Polygon rotation (deg)')
        axo.set_ylabel(f'Model-center offset ({dist_unit_label()})')
        axo.set_title(f'Model-center offset from centroids vs polygon rotation ({src})')
        axo.grid(True, linestyle=':', alpha=0.6)
        if legend_handles:
            # Cap at 2 columns so the 4-entry 'both' legend wraps into 2 rows
            # below the plot instead of overflowing the figure width.
            figo.legend(legend_handles, legend_labels, loc='outside lower center',
                        ncol=min(len(legend_handles), 2), fontsize=8)

    # ---- Fill comparison: mean semi-axis lengths +/- SD, Filled vs Unfilled - #
    if fill == 'both' and {True, False} <= set(fills_present):
        _fig_axis_length_bars(
            usable,
            title=f'Mean model axis lengths (±SD): Filled vs Unfilled ({src})')


def _fig_axis_length_bars(usable, title='Mean model axis lengths (±SD): Filled vs Unfilled',
                          by_subject=False):
    """Grouped bar chart of mean semi-major/semi-minor (+/-SD), Filled vs Unfilled.

    Laid out like the offset bar chart: one column per axis (Major / Minor),
    with Filled vs Unfilled side-by-side bars, a Condition legend, and a black
    asterisk over any axis where the two conditions differ significantly.
    """
    fills = [f for f in (True, False) if any(r['is_filled'] == f for r in usable)]
    if len(fills) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    metrics = [('Major', 'semi_major'), ('Minor', 'semi_minor')]
    _draw_grouped_bars(
        ax, usable, fills, metrics,
        value_fn=lambda r, key: to_distance(r['stats'][key]),
        ylabel=f'Mean semi-axis length ({dist_unit_label()})', title=title,
        by_subject=by_subject)


# --------------------------------------------------------------------------- #
# All-data analytics (PLOT_MODE == 'analytics') -- pooled over EVERY trial
# --------------------------------------------------------------------------- #
def _read_session_efix(asc_file_path):
    """Read a session's ASC once: return (time_offset, [(start_ms, x, y), ...]).

    time_offset comes from the first TRIAL_START; all EFIX rows are collected and
    aligned later. Returns (None, []) if no usable TRIAL_START is found.
    """
    time_offset = None
    efix = []
    with open(asc_file_path, 'r') as f:
        for line in f:
            if time_offset is None and 'TRIAL_START' in line:
                parts = line.split()
                try:
                    time_offset = float(parts[1]) / 1000.0 - float(parts[3])
                except (IndexError, ValueError):
                    pass
            elif line.startswith('EFIX'):
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        efix.append((float(parts[2]), float(parts[5]),
                                     float(parts[6])))
                    except ValueError:
                        pass
    return time_offset, efix


def collect_all_stats(root_dir, aggregate=MULTI_AGGREGATE, by_subject=False):
    """
    Fit a 2D Gaussian to every auto-polygon trial and return the parameter
    records (no plotting). Each record has: sides / rotation / step / is_filled /
    base_centroid / actual_centroid / stats.

    by_subject = False (default): fixations are pooled across all sessions, so
    each record is one trial (one ellipse fit on every participant's gaze).

    by_subject = True: one record PER (participant, trial) for DISTANCE/offset
    analyses only. 'stats' holds just the subject's arithmetic gaze centroid
    (mean X/Y of that trial's fixations) -- no ellipse is fit -- and a
    'subject_id' field is added, so downstream figures can average within subject
    (repeated-measures). Orientation/ellipse-shape metrics are NOT produced here;
    those always use the pooled records.

    Each session's ASC and CSV are read ONCE and filtered in memory, so this
    stays fast. Window, saccade-latency, and boundary filtering match
    extract_valid_fixations.
    """
    sessions = find_session_folders(root_dir) if aggregate else [EXAMPLE_SESSION]
    if not sessions:
        return [], 0

    fill_map = build_fill_map()

    def _is_filled(ti, params):
        combo = fill_map.get(ti)
        if (combo and combo['sides'] == params['sides']
                and combo['step'] == params['step']
                and combo['rotation'] == params['rotation']):
            return combo['is_filled']
        return None

    pooled = {}            # trial_index -> accumulator (cross-subject pooling)
    subject_records = []   # one per (subject, trial) when by_subject
    n_used = 0
    for sess in sessions:
        try:
            asc_path = _resolve_asc_path(sess)
        except FileNotFoundError:
            continue
        time_offset, efix = _read_session_efix(asc_path)
        if time_offset is None:
            continue
        n_used += 1
        # Participant id (root/participant_X/session_Y -> participant_X).
        subject_id = os.path.basename(os.path.dirname(sess))

        df = pd.read_csv(os.path.join(sess, 'trials.csv'))
        df = df[df['trial_index'].notna()].copy()
        df['trial_index'] = df['trial_index'].astype(int)
        for _i, row in df.iterrows():
            params = classify_trial(row)
            if params is None:
                continue
            ti = int(row['trial_index'])
            stim_on = float(row['stim_on_ts_s'])
            stim_off = float(row['stim_off_ts_s'])
            window_start = stim_on + STIM_ONSET_LATENCY_S
            xs, ys = [], []
            for start_ms, x, y in efix:
                aligned = start_ms / 1000.0 - time_offset
                if window_start <= aligned <= stim_off:
                    xs.append(x)
                    ys.append(y)
            polygon = _place_polygon(
                parse_polygon_str(row['polygon_points_imgspace']),
                (float(row['stim_top_left_x_px']),
                 float(row['stim_top_left_y_px'])))
            base_centroid = (float(row['stim_center_x_px']),
                             float(row['stim_center_y_px']))

            if by_subject:
                # Distances use the subject's arithmetic gaze centroid for this
                # trial (mean X/Y of the fixations) -- NOT a trial-level ellipse
                # fit -- so every trial with >=1 fixation is kept.
                fx, fy = _filter_by_boundary_distance(
                    xs, ys, polygon,
                    FIXATION_BOUNDARY_DIST_PX, FIXATION_BOUNDARY_MODE)
                if not fx:
                    continue
                gaze_mean = (float(np.mean(fx)), float(np.mean(fy)))
                subject_records.append({
                    'trial_index': ti, 'subject_id': subject_id,
                    'sides': params['sides'], 'rotation': params['rotation'],
                    'step': params['step'], 'is_filled': _is_filled(ti, params),
                    'base_centroid': base_centroid,
                    'actual_centroid': _actual_centroid(polygon),
                    'stats': {'mean': gaze_mean},   # only the centre is defined
                })
            else:
                if ti not in pooled:
                    pooled[ti] = {
                        'x': [], 'y': [], 'polygon': polygon,
                        'base_centroid': base_centroid,
                        'sides': params['sides'], 'rotation': params['rotation'],
                        'step': params['step'],
                    }
                pooled[ti]['x'].extend(xs)
                pooled[ti]['y'].extend(ys)

    if by_subject:
        return subject_records, n_used

    records = []
    for ti, d in pooled.items():
        # Spatial boundary filter on the pooled fixations (same as single-trial).
        x, y = _filter_by_boundary_distance(
            d['x'], d['y'], d['polygon'],
            FIXATION_BOUNDARY_DIST_PX, FIXATION_BOUNDARY_MODE)
        records.append({
            'trial_index': ti, 'sides': d['sides'], 'rotation': d['rotation'],
            'step': d['step'],
            'is_filled': _is_filled(ti, {'sides': d['sides'],
                                         'step': d['step'],
                                         'rotation': d['rotation']}),
            'base_centroid': d['base_centroid'],
            'actual_centroid': _actual_centroid(d['polygon']),
            'stats': fit_gaussian(x, y, ref_point=d['base_centroid']),
        })
    return records, n_used


def _fills_to_plot(fill):
    return {'both': [True, False], 'filled': [True],
            'unfilled': [False]}.get(fill, [True, False])


def _cell_values(records, predicate, value_fn, by_subject):
    """Collect one numeric value per data point for a condition cell.

    Trials are selected by `predicate(record)`; `value_fn(record)` returns the
    per-trial metric (or None). With `by_subject`, the metrics are first averaged
    within each subject (so the returned values are per-subject means); otherwise
    every matching trial contributes one value.
    """
    matched = [r for r in records if predicate(r)]
    if not by_subject:
        return [v for v in (value_fn(r) for r in matched) if v is not None]
    per_subject = {}
    for r in matched:
        v = value_fn(r)
        if v is not None:
            per_subject.setdefault(r.get('subject_id'), []).append(v)
    return [float(np.mean(vs)) for vs in per_subject.values() if vs]


def _draw_grouped_bars(ax, usable, fills_to_plot, metrics, value_fn, ylabel,
                       title, by_subject=False):
    """Draw a Filled/Unfilled grouped bar chart (mean +/- SD) with significance.

    `metrics` is a list of (column_label, key); `value_fn(record, key)` returns
    one numeric value per trial (or None). Bars are dodged by condition; when
    both Filled and Unfilled are present, a Welch t-test compares them within
    each category and a black asterisk is drawn above any significant category.
    With `by_subject`, every value is first averaged within each subject.
    """
    xpos = np.arange(len(metrics))
    width = 0.8 / len(fills_to_plot)
    raw, tops = {}, {}
    for i, isf in enumerate(fills_to_plot):
        means, sds = [], []
        for mi, (_label, key) in enumerate(metrics):
            vals = _cell_values(usable, lambda r, _isf=isf: r['is_filled'] == _isf,
                                lambda r, _k=key: value_fn(r, _k), by_subject)
            raw[(mi, isf)] = vals
            m = float(np.mean(vals)) if vals else 0.0
            s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            means.append(m)
            sds.append(s)
            tops[mi] = max(tops.get(mi, 0.0), m + s)
        off = (i - (len(fills_to_plot) - 1) / 2.0) * width
        ax.bar(xpos + off, means, width, yerr=sds, color=FILL_COLORS[isf],
               alpha=0.6, edgecolor='black', linewidth=0.8, capsize=6,
               error_kw=dict(ecolor='black', elinewidth=1.2, capthick=1.2),
               label=_fill_word(isf), zorder=2)
        if SHOW_RAW_POINTS:
            for mi in range(len(metrics)):
                _scatter_raw(ax, xpos[mi] + off, raw[(mi, isf)],
                             _shade(FILL_COLORS[isf], 0.6), jitter=width * 0.28)
    ax.set_xticks(xpos)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylabel(ylabel)

    if SHOW_SIGNIFICANCE and {True, False} <= set(fills_to_plot):
        span = max(tops.values()) if tops else 1.0
        pad = 0.04 * span if span > 0 else 1.0
        for mi in range(len(metrics)):
            if is_significant(welch_p(raw.get((mi, True)), raw.get((mi, False)))):
                _annotate_star(ax, xpos[mi], tops[mi] + pad, 'black')
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    if len(fills_to_plot) > 1:
        ax.legend(title='Condition', loc='upper left')

    # --- print the p-value(s) for this graph ---
    print(f"[p] {title}")
    if {True, False} <= set(fills_to_plot):
        for mi, (label, _k) in enumerate(metrics):
            p = welch_p(raw.get((mi, True)), raw.get((mi, False)))
            print(f"    {label.replace(chr(10), ' ')}: Filled vs Unfilled "
                  f"(Welch t)  p={_fmt_p(p)}")
    elif len(metrics) == 2:
        isf = fills_to_plot[0]
        p = paired_p(raw.get((0, isf)), raw.get((1, isf)))
        c0 = metrics[0][0].replace(chr(10), ' ')
        c1 = metrics[1][0].replace(chr(10), ' ')
        print(f"    {c0} vs {c1} (paired t)  p={_fmt_p(p)}")


def _fig_offset_bars(usable, fills_to_plot, by_subject=False):
    """Grouped bars: mean ellipse-center offset from each centroid (+/-SD)."""
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    metrics = [('Original\ncentroid', 'base_centroid'),
               ('Center\nof mass', 'actual_centroid')]

    def value_fn(r, key):
        return to_distance(_euclidean(r[key], r['stats']['mean']))

    title = 'All trials — mean model-center offset from centroids (±SD)'
    _draw_grouped_bars(
        ax, usable, fills_to_plot, metrics, value_fn,
        ylabel=f'Mean model-center offset ({dist_unit_label()})', title=title,
        by_subject=by_subject)


def _scatter_raw(ax, x_center, values, color, jitter=0.0, zorder=4):
    """Overlay individual-trial values as a jittered strip at `x_center`."""
    if not values:
        return
    rng = np.random.default_rng(0)      # deterministic jitter
    if jitter > 0:
        xs = x_center + rng.uniform(-jitter, jitter, size=len(values))
    else:
        xs = np.full(len(values), x_center)
    ax.scatter(xs, values, s=12, color=color, alpha=0.35, edgecolors='none',
               zorder=zorder)


def _shade(color, factor):
    """Return `color` scaled toward black by `factor` (1.0 = unchanged)."""
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def _circular_mean_deg(angles):
    """Circular mean (in degrees, [0, 360)) of a list of angles in degrees."""
    s = sum(math.sin(math.radians(a)) for a in angles)
    c = sum(math.cos(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(s, c)) % 360.0


def _circular_std_deg(angles):
    """Circular standard deviation (deg) of angles in degrees (Mardia).

    Returns 0 for <2 samples; larger values mean more angular dispersion.
    """
    if len(angles) < 2:
        return 0.0
    s = sum(math.sin(math.radians(a)) for a in angles)
    c = sum(math.cos(math.radians(a)) for a in angles)
    R = math.hypot(s, c) / len(angles)          # mean resultant length
    R = max(min(R, 1.0), 1e-6)                   # clamp to keep the log finite
    return math.degrees(math.sqrt(-2.0 * math.log(R)))


def _fig_offset_vs(usable, fills_to_plot, xkey, xlabel, title, dodge,
                   by_subject=False):
    """Line plot: mean ellipse-center offset (+/-SD) vs `xkey` (step or rotation).

    Color = fill; linestyle = which centroid (base solid, center of mass dashed,
    drawn in a darker shade). Series are dodged along X by `dodge` so the error
    bars sit side-by-side rather than overlapping. With `by_subject`, each x/fill
    cell's values are subject means.
    """
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    metrics = [('Offset from original centroid', 'base_centroid', '-', 1.0),
               ('Offset from center of mass', 'actual_centroid', '--', 0.6)]
    xvals = sorted({r[xkey] for r in usable if r[xkey] is not None})

    # Asterisk colors encode the centroid being compared: bright orange for the
    # original centroid, dark orange for the center of mass.
    metric_star_colors = ['#FF8C00', '#8A4C14']

    series = []
    samples = {}   # (metric_index, is_filled, x) -> list of per-trial offsets
    tops = {}      # (metric_index, is_filled, x) -> mean + SD (point top)
    for isf in fills_to_plot:
        cond = f" ({_fill_word(isf)})" if len(fills_to_plot) > 1 else ""
        for mi, (name, key, ls, shade) in enumerate(metrics):
            xs, ys, es = [], [], []
            for xv in xvals:
                vals = _cell_values(
                    usable,
                    lambda r, _isf=isf, _xv=xv: (r['is_filled'] == _isf
                                                 and r[xkey] == _xv),
                    lambda r, _k=key: to_distance(
                        _euclidean(r[_k], r['stats']['mean'])),
                    by_subject)
                if not vals:
                    continue
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                xs.append(xv)
                ys.append(m)
                es.append(s)
                samples[(mi, isf, xv)] = vals
                tops[(mi, isf, xv)] = m + s
            if not xs:
                continue
            series.append((xs, ys, es, _shade(FILL_COLORS[isf], shade), ls,
                           f"{name}{cond}", mi, isf))

    n = len(series)
    handles, labels = [], []
    for idx, (xs, ys, es, color, ls, label, mi, isf) in enumerate(series):
        shift = (idx - (n - 1) / 2.0) * dodge
        xs_d = [x + shift for x in xs]
        ax.errorbar(xs_d, ys, yerr=es, color=color, linestyle=ls, marker='o',
                    capsize=4, elinewidth=1.2, alpha=0.7)
        handles.append(Line2D([0], [0], color=color, linestyle=ls, linewidth=2))
        labels.append(label)
        if SHOW_RAW_POINTS:
            jitter = abs(dodge) * 0.35 if dodge else 0.0
            for xd, xv in zip(xs_d, xs):
                _scatter_raw(ax, xd, samples[(mi, isf, xv)], color, jitter=jitter)

    # Significance asterisks. With both conditions shown, compare Filled vs
    # Unfilled within each centroid (asterisk colored by centroid). With a single
    # condition, compare the two centroids against each other (paired, since both
    # come from the same trials) and use a black asterisk.
    if SHOW_SIGNIFICANCE and tops:
        all_tops = list(tops.values())
        yspan = (max(all_tops) - min(all_tops)) if len(all_tops) > 1 else 1.0
        star_step = max(0.03 * yspan, 1e-6)
        if {True, False} <= set(fills_to_plot):
            for mi in range(len(metrics)):
                for xv in xvals:
                    a = samples.get((mi, True, xv))
                    b = samples.get((mi, False, xv))
                    if is_significant(welch_p(a, b)):
                        base_top = max(tops.get((mi, True, xv), 0.0),
                                       tops.get((mi, False, xv), 0.0))
                        # Stagger the two centroids vertically to avoid overlap.
                        y_top = base_top + star_step * (1.0 + mi)
                        _annotate_star(ax, xv, y_top, metric_star_colors[mi])
        else:
            isf = fills_to_plot[0]
            for xv in xvals:
                a = samples.get((0, isf, xv))   # original centroid
                b = samples.get((1, isf, xv))   # center of mass
                if is_significant(paired_p(a, b)):
                    base_top = max(tops.get((0, isf, xv), 0.0),
                                   tops.get((1, isf, xv), 0.0))
                    _annotate_star(ax, xv, base_top + star_step, 'black')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'Mean model-center offset ({dist_unit_label()})')
    ax.set_title(title)
    if xvals:
        ax.set_xticks(xvals)
    ax.grid(True, linestyle=':', alpha=0.6)
    if handles:
        fig.legend(handles, labels, loc='outside lower center',
                   ncol=min(len(handles), 2), fontsize=8)

    # --- print the p-value(s) for this graph ---
    # Both conditions: Filled vs Unfilled per centroid, at each x. Single
    # condition: the two centroids vs each other (paired), at each x.
    print(f"[p] {title}")
    if {True, False} <= set(fills_to_plot):
        for mi, (name, _k, _ls, _sh) in enumerate(metrics):
            for xv in xvals:
                p = welch_p(samples.get((mi, True, xv)),
                            samples.get((mi, False, xv)))
                print(f"    {name} @ {xlabel.split()[0].lower()}={xv}: "
                      f"Filled vs Unfilled  p={_fmt_p(p)}")
    else:
        isf = fills_to_plot[0]
        for xv in xvals:
            p = paired_p(samples.get((0, isf, xv)), samples.get((1, isf, xv)))
            print(f"    original centroid vs center of mass @ "
                  f"{xlabel.split()[0].lower()}={xv} (paired)  p={_fmt_p(p)}")


def _add_regression(ax, xs, ys, color, x_range):
    """Draw an OLS trendline + 95% CI band over `x_range`; return a stats string.

    The 95% confidence interval for the mean response is computed analytically
    (Student-t), so no seaborn dependency is needed. Returns 'r / R^2 / p' text,
    or None when there are too few points.
    """
    if len(xs) < 2:
        return None
    xa = np.asarray(xs, dtype=float)
    ya = np.asarray(ys, dtype=float)
    lr = linregress(xa, ya)
    xl = np.linspace(x_range[0], x_range[1], 100)
    yl = lr.slope * xl + lr.intercept
    ax.plot(xl, yl, color=color, linestyle='-', linewidth=1.6, alpha=0.9,
            zorder=1)

    n = len(xa)
    if n >= 3:
        resid = ya - (lr.slope * xa + lr.intercept)
        s_err = math.sqrt(np.sum(resid ** 2) / (n - 2))
        xmean = xa.mean()
        sxx = np.sum((xa - xmean) ** 2)
        if sxx > 0:
            se = s_err * np.sqrt(1.0 / n + (xl - xmean) ** 2 / sxx)
            tcrit = t_dist.ppf(0.975, n - 2)
            ax.fill_between(xl, yl - tcrit * se, yl + tcrit * se,
                            color=color, alpha=0.15, zorder=0)

    return (f"r = {lr.rvalue:.2f},  R² = {lr.rvalue ** 2:.2f},  "
            f"p = {lr.pvalue:.3f}")


def _fig_orientation_vs_rotation(usable, fills_to_plot, by_subject=False):
    """Line plot: mean ellipse orientation (circular mean) vs polygon rotation.

    With `by_subject`, each rotation/fill cell is first reduced to one circular
    mean per subject, so the plotted mean/SD and raw points are subject-level.
    """
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    rotations = sorted({r['rotation'] for r in usable if r['rotation'] is not None})
    all_y = []
    # Slightly stagger the conditions on X so their error bars do not overlap.
    dodge = 2.0
    x_range = (min(rotations), max(rotations)) if rotations else (0, 1)
    reg_notes = []          # (text, color) per condition for the stats box
    for i, isf in enumerate(fills_to_plot):
        xs, ys, es, raws = [], [], [], []
        for rot in rotations:
            sel = [r['stats']['angle_deg_360'] for r in usable
                   if r['is_filled'] == isf and r['rotation'] == rot]
            if by_subject:
                # One circular mean per subject, then aggregate across subjects.
                per_subj = {}
                for r in usable:
                    if r['is_filled'] == isf and r['rotation'] == rot:
                        per_subj.setdefault(r.get('subject_id'), []).append(
                            r['stats']['angle_deg_360'])
                angs = [_circular_mean_deg(a) for a in per_subj.values() if a]
            else:
                angs = sel
            if not angs:
                continue
            xs.append(rot)
            ys.append(_circular_mean_deg(angs))
            es.append(_circular_std_deg(angs))   # angular variability (deg)
            raws.append(angs)                    # individual-trial orientations
        offset0 = 0.0
        if len(ys) > 1:
            # An ellipse axis is only defined modulo 180 deg, so unwrap with a
            # 180 deg (pi rad) period: this removes the symmetry-induced "drop"
            # (e.g. past 135 deg) and keeps the orientation trend continuous.
            ys = list(np.degrees(np.unwrap(np.radians(ys), period=np.pi)))
            # Align the orientation axis with the polygon-rotation axis: shift the
            # whole (already continuous) curve so rotation 0 deg maps to 0 deg
            # orientation. A constant offset preserves continuity -- no new phase
            # jumps -- it only re-bases where the trend starts.
            base_idx = xs.index(0) if 0 in xs else 0
            offset0 = ys[base_idx]
            ys = [y - offset0 for y in ys]
        # Include the error-bar extent so the Y limits never clip the SD whiskers.
        all_y.extend([y - e for y, e in zip(ys, es)])
        all_y.extend([y + e for y, e in zip(ys, es)])
        label = _fill_word(isf) if len(fills_to_plot) > 1 else 'Orientation θ'
        shift = (i - (len(fills_to_plot) - 1) / 2.0) * dodge
        xs_d = [x + shift for x in xs]
        # Darker shade for the data points + SD whiskers so they stand out from
        # the (lighter) regression line drawn in the base fill color.
        point_color = _shade(FILL_COLORS[isf], 0.6)
        ax.errorbar(xs_d, ys, yerr=es, fmt='o', color=point_color,
                    capsize=4, elinewidth=1.2, alpha=0.9, label=label)

        if SHOW_RAW_POINTS:
            for k, xd in enumerate(xs_d):
                # Map each raw orientation into the aligned frame: subtract the
                # baseline offset, then wrap to within +-90 deg of this point's
                # mean (orientation is axial, i.e. defined mod 180 deg).
                pts = []
                for a in raws[k]:
                    v = a - offset0
                    while v - ys[k] > 90.0:
                        v -= 180.0
                    while v - ys[k] < -90.0:
                        v += 180.0
                    pts.append(v)
                all_y.extend(pts)
                _scatter_raw(ax, xd, pts, point_color, jitter=dodge * 0.4)

        # Linear regression trendline + 95% CI band, with stats for the box.
        note = _add_regression(ax, xs, ys, FILL_COLORS[isf], x_range)
        if note is not None:
            prefix = f"{_fill_word(isf)}: " if len(fills_to_plot) > 1 else ""
            reg_notes.append((prefix + note, FILL_COLORS[isf]))

    ax.set_xlabel('Polygon rotation (deg)', fontsize=11)
    ax.set_ylabel('Mean Model orientation θ (deg, baseline-aligned)', fontsize=11)
    ax.set_title('All trials — mean Model orientation vs polygon rotation',
                 fontsize=13)
    ax.tick_params(labelsize=10)
    if rotations:
        ax.set_xticks(rotations)
    if all_y:
        # Fit the Y axis to the baseline-aligned data (starts at 0 deg for the
        # rotation-0 baseline). Dynamic limits so nothing is clipped/dropped.
        lo, hi = min(all_y), max(all_y)
        pad = max(10.0, 0.08 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
    ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.6)

    # Publication-ready stats box (Pearson r, R^2, p) in the lower-right corner.
    y0 = 0.03
    for text, color in reversed(reg_notes):
        ax.text(0.975, y0, text, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, color=color,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor='0.6', alpha=0.9))
        y0 += 0.075

    if len(fills_to_plot) > 1:
        ax.legend(title='Condition', loc='upper left')

    # --- print the p-value(s) for this graph (orientation~rotation regression) ---
    print('[p] All trials — mean Model orientation vs polygon rotation')
    if reg_notes:
        for text, _color in reg_notes:
            print(f"    linear regression: {text}")
    else:
        print('    linear regression: n/a (too few points)')


def _fig_center_bias(usable, fills_to_plot, by_subject=False):
    """Evaluate whether the gaze deviation is biased toward the screen center.

    All geometry is in full-screen pixel coordinates (0..SCREEN_W x 0..SCREEN_H).
    For each trial, with the polygon's geometric centroid as the common origin:

        screen_vec = screen_center        - polygon_centroid
        gaze_vec   = ellipse_center(gaze) - polygon_centroid
        projection = gaze_vec . unit(screen_vec)

    A positive projection means the gaze center is displaced TOWARD the screen
    center; negative means away from it. A one-sample t-test vs 0 tests whether
    the mean projection differs from zero (i.e. a systematic center bias).

    Two panels: (left) the gaze vectors drawn in screen space with the screen
    center marked, colored by projection sign; (right) the distribution of the
    scalar projections with the per-condition mean and t-test annotation.
    """
    screen_center = np.array([SCREEN_W / 2.0, SCREEN_H / 2.0])

    # Stimuli are shown on a 3x3 grid of positions; the requirement is to use
    # only the OFF-center grid points. Identify the center grid point (the stim
    # placement nearest the screen center) from the recorded stim centers and
    # drop every trial shown there.
    grid_pts = {(round(r['base_centroid'][0]), round(r['base_centroid'][1]))
                for r in usable if r['base_centroid'] is not None}
    center_grid = (min(grid_pts, key=lambda p: math.hypot(
        p[0] - screen_center[0], p[1] - screen_center[1]))
        if grid_pts else None)
    grid_tol = 5.0   # px; grid spacing is ~144 px, so this isolates one point

    rows = []
    n_excluded = 0
    for r in usable:
        bc = r['base_centroid']              # stim placement (grid point)
        pc = r['actual_centroid']            # polygon (geometric) centroid
        gaze = r['stats']['mean']            # fitted ellipse center
        if bc is None or pc is None or gaze is None:
            continue
        # Exclude the center grid point (stimulus centered on the screen middle).
        if center_grid is not None and math.hypot(
                bc[0] - center_grid[0], bc[1] - center_grid[1]) < grid_tol:
            n_excluded += 1
            continue
        pc = np.asarray(pc, dtype=float)
        gaze = np.asarray(gaze, dtype=float)
        svec = screen_center - pc
        slen = float(np.hypot(*svec))
        if slen < 1e-6:                      # polygon already at screen center
            continue
        gvec = gaze - pc
        proj = float(np.dot(gvec, svec) / slen)   # scalar projection (px)
        rows.append(dict(pc=pc, gvec=gvec, proj=proj, is_filled=r['is_filled'],
                         subject_id=r.get('subject_id')))
    if not rows:
        print('[center-bias] no usable off-center trials.')
        return
    print(f"[center-bias] {len(rows)} off-center trials "
          f"(excluded {n_excluded} at center grid {center_grid}).")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6),
                                   constrained_layout=True)

    # ---- Left: gaze vectors in full-screen space ------------------------- #
    scale = 2.0                              # exaggerate gaze vectors for clarity
    for d in rows:
        col = 'tab:green' if d['proj'] >= 0 else 'tab:red'
        tip = d['pc'] + d['gvec'] * scale
        ax1.annotate('', xy=tip, xytext=d['pc'],
                     arrowprops=dict(arrowstyle='->', color=col, lw=1.0,
                                     alpha=0.7))
        ax1.scatter(*d['pc'], s=8, color='0.4', zorder=3)
    ax1.scatter(*screen_center, marker='*', s=240, color='black', zorder=6)
    ax1.set_xlim(0, SCREEN_W)
    ax1.set_ylim(SCREEN_H, 0)               # inverted Y: (0,0) at top-left
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('X (px)', fontsize=11)
    ax1.set_ylabel('Y (px)', fontsize=11)
    ax1.set_title('Gaze displacement vs screen-center direction', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.5, linewidth=0.6)
    ax1.legend(handles=[
        Line2D([0], [0], color='tab:green', lw=2, label='Toward center (proj > 0)'),
        Line2D([0], [0], color='tab:red', lw=2, label='Away from center (proj < 0)'),
        Line2D([0], [0], marker='*', color='black', lw=0, markersize=12,
               label='Screen center')], fontsize=8, loc='best')
    ax1.text(0.02, 0.02, f'gaze arrows ×{scale:g}', transform=ax1.transAxes,
             fontsize=8, color='0.4')

    # ---- Right: distribution of scalar projections + t-test vs 0 --------- #
    # With by_subject, reduce each fill's projections to one mean per subject.
    def _projections(isf):
        return _cell_values(rows, lambda d, _isf=isf: d['is_filled'] == _isf,
                            lambda d: d['proj'], by_subject)

    proj_all = [d['proj'] for d in rows]
    bins = (np.linspace(min(proj_all), max(proj_all), 20)
            if len(proj_all) > 1 else 10)
    notes = []
    for isf in fills_to_plot:
        pr = _projections(isf)
        if not pr:
            continue
        color = FILL_COLORS[isf]
        lab = _fill_word(isf) if len(fills_to_plot) > 1 else 'Projection'
        ax2.hist(pr, bins=bins, color=color, alpha=0.5, label=lab)
        m = float(np.mean(pr))
        ax2.axvline(m, color=_shade(color, 0.6), linestyle='--', linewidth=1.5)
        p = float(ttest_1samp(pr, 0.0).pvalue) if len(pr) > 1 else float('nan')
        bias = 'toward' if m > 0 else 'away'
        unit = 'subjects' if by_subject else 'trials'
        notes.append((f'{_fill_word(isf)}: mean={m:.1f}px ({bias}), '
                      f'p={p:.3f} (n={len(pr)} {unit})', color))
    ax2.axvline(0, color='black', linewidth=1.0)
    ax2.set_xlabel('Scalar projection toward screen center (px)', fontsize=11)
    ax2.set_ylabel('Subject count' if by_subject else 'Trial count', fontsize=11)
    ax2.set_title('Center-bias of gaze displacement', fontsize=12)
    ax2.grid(True, axis='y', linestyle=':', alpha=0.5, linewidth=0.6)
    if len(fills_to_plot) > 1:
        ax2.legend(fontsize=8, loc='upper right')
    y0 = 0.97
    for text, color in notes:
        ax2.text(0.02, y0, text, transform=ax2.transAxes, va='top', ha='left',
                 fontsize=9, color=color,
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                           edgecolor='0.6', alpha=0.9))
        y0 -= 0.085

    # --- print the p-value(s) for this graph (one-sample t-test vs 0) ---
    print('[p] Center-bias of gaze displacement (one-sample t-test of '
          'projection vs 0)')
    for text, _color in notes:
        print(f"    {text}")


def plot_all_data_statistics(root_dir, fill=ANALYTICS_FILL,
                             aggregate=MULTI_AGGREGATE,
                             sides=ANALYTICS_SIDES,
                             rotations=ANALYTICS_ROTATIONS,
                             steps=ANALYTICS_STEPS,
                             use_subject_averages=USE_SUBJECT_AVERAGES):
    """
    Analytical graphs computed over the trials (no polygon/fixation panels):
        1. ellipse axis lengths, Filled vs Unfilled (always both).
        2. ellipse-center offset from original centroid & center of mass.
        3. mean ellipse-center offset vs step size.
        4. mean ellipse-center offset vs polygon rotation.
        5. mean ellipse orientation vs polygon rotation.
        6. screen-center bias of the gaze displacement (if SHOW_CENTER_BIAS).
    Graphs 2-5 split by fill when `fill == 'both'`, else show one condition.

    `rotations` / `steps` (lists or None) restrict the dataset to those polygon
    rotations / stretch steps. This filters by stimulus geometry only -- all
    participants are always pooled.
    """
    # Optional geometry filters (not by participant). Accept a single value or a
    # list/tuple/set; None means no filtering.
    def _as_set(v):
        if v is None:
            return None
        return set(v) if isinstance(v, (list, tuple, set)) else {v}

    sides_set, rot_set, step_set = (_as_set(sides), _as_set(rotations),
                                    _as_set(steps))

    def _filt(recs):
        if sides_set is not None:
            recs = [r for r in recs if r['sides'] in sides_set]
        if rot_set is not None:
            recs = [r for r in recs if r['rotation'] in rot_set]
        if step_set is not None:
            recs = [r for r in recs if r['step'] in step_set]
        return recs

    # Pooled records (one ellipse per trial across all participants). These drive
    # the orientation and ellipse-shape figures ALWAYS, regardless of the
    # subject-averages option.
    pooled, n_sessions = collect_all_stats(root_dir, aggregate=aggregate)
    pooled = _filt([r for r in pooled
                    if r['stats'] is not None and r['is_filled'] is not None])
    if not pooled:
        print('[all-data] no trials after filtering; nothing to plot.')
        return

    # Distance/offset records. With subject averages, use per-(subject, trial)
    # arithmetic gaze centroids (no ellipse fit); otherwise reuse the pooled set.
    bs = use_subject_averages
    if bs:
        dist_recs, _ = collect_all_stats(root_dir, aggregate=aggregate,
                                         by_subject=True)
        dist_recs = _filt([r for r in dist_recs if r['is_filled'] is not None])
    else:
        dist_recs = pooled

    filt = []
    if sides is not None:
        filt.append(f"sides={sides}")
    if rotations is not None:
        filt.append(f"rotations={rotations}")
    if steps is not None:
        filt.append(f"steps={steps}")
    filt_note = f" ({', '.join(filt)})" if filt else ""
    dist_note = (f"; {len(dist_recs)} per-subject trial centroids for distances"
                 if bs else "")
    print(f"[all-data] {len(pooled)} pooled ellipses (orientation/shape) from "
          f"{n_sessions} session(s){dist_note}{filt_note}.")

    fills_to_plot = _fills_to_plot(fill)
    # Orientation & ellipse parameters: ALWAYS pooled across all trials.
    _fig_axis_length_bars(pooled, title='All trials — mean model axis lengths (±SD)',
                          by_subject=False)
    _fig_orientation_vs_rotation(pooled, fills_to_plot, by_subject=False)
    # Distances & offsets (incl. center bias): subject-level when requested.
    _fig_offset_bars(dist_recs, fills_to_plot, by_subject=bs)
    _fig_offset_vs(dist_recs, fills_to_plot, 'step', 'Stretch step level',
                   'All trials — mean offset from centroids vs step size',
                   dodge=0.08, by_subject=bs)
    _fig_offset_vs(dist_recs, fills_to_plot, 'rotation', 'Polygon rotation (deg)',
                   'All trials — mean offset from centroids vs polygon rotation',
                   dodge=4.0, by_subject=bs)
    if SHOW_CENTER_BIAS:
        _fig_center_bias(dist_recs, fills_to_plot, by_subject=bs)
    plt.show()


# --------------------------------------------------------------------------- #
# Mode C: several trials side by side
# --------------------------------------------------------------------------- #
def plot_multi_trials(root_dir, group_by=MULTI_GROUP_BY, sides=MULTI_SIDES,
                      step=MULTI_STEP, rotation=MULTI_ROTATION,
                      aggregate=MULTI_AGGREGATE, fill=MULTI_FILL,
                      fixation_style=FIXATION_STYLE,
                      show_actual=SHOW_ACTUAL_CENTROID,
                      show_base=SHOW_BASE_CENTROID,
                      show_ellipse=SHOW_GAUSSIAN_ELLIPSE,
                      ellipse_only=ELLIPSE_ONLY,
                      show_stats_plot=SHOW_STATS_PLOT):
    """
    Plot several related trials side by side in a single figure.

    group_by == 'rotations' : every rotation of the polygon (sides, step).
    group_by == 'variations': every variation/step of `sides` vertices at the
                              fixed `rotation`.

    fill == 'filled' | 'unfilled' | 'both' restricts to image-filled polygons,
    plain (black) polygons, or both. Each panel title notes its fill state.

    Fixations are pooled across all sessions when `aggregate` is True, otherwise
    only EXAMPLE_SESSION is used.
    """
    sessions = find_session_folders(root_dir) if aggregate else [EXAMPLE_SESSION]
    if not sessions:
        print(f"No session folders found under {root_dir}")
        return

    # Use one session's CSV to enumerate trials (identical across participants).
    selected = select_trials(EXAMPLE_SESSION, group_by, sides,
                              step=step, rotation=rotation, fill=fill)
    if not selected:
        print(f"No trials matched group_by={group_by!r}, sides={sides}, "
              f"step={step}, rotation={rotation}, fill={fill!r}.")
        return

    # Lay out the panels. With fill='both', use one row per condition (filled on
    # top, unfilled below); otherwise a single row.
    if fill == 'both':
        rows = [[p for p in selected if p['is_filled'] is True],
                [p for p in selected if p['is_filled'] is False]]
        rows = [r for r in rows if r]          # drop an absent condition
    else:
        rows = [selected]
    nrows = len(rows)
    ncols = max(len(r) for r in rows)

    # constrained_layout (not tight_layout) cleanly makes room for the per-panel
    # colorbars without the "Axes not compatible with tight_layout" warning.
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows),
                             squeeze=False, constrained_layout=True)
    # Tighten the vertical gaps so the suptitle and the bottom legend sit close
    # to the panels.
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.01,
                                    wspace=0.02, hspace=0.02)

    color = '#E74C3C' if aggregate else '#2ECC71'
    alpha = 0.25 if aggregate else 0.85
    varying = 'rotation' if group_by == 'rotations' else 'step'

    def fill_label(is_filled):
        return {True: 'filled', False: 'unfilled'}.get(is_filled, 'fill unknown')

    # Pre-collect every panel's data so we can derive ONE shared window before
    # drawing. Keyed by trial_index (unique per panel, incl. filled/unfilled).
    collected = {}
    for params in selected:
        ti = params['trial_index']
        x, y, polygon, base_centroid, _n = collect_fixations(ti, sessions)
        collected[ti] = (x, y, polygon, base_centroid)

    # Center every polygon on its own actual centroid and draw it in
    # centroid-RELATIVE coordinates, so every panel shares the *identical* axis
    # range [-half, +half] (same tick numbers, not just the same span). `half`
    # is driven by the largest content across panels -- each polygon's farthest
    # vertex AND, when drawn, the fitted confidence ellipse (so a big-STD ellipse
    # is never clipped) -- keeping every panel centered, fully shown, and at one
    # uniform scale. (VIEW == 'screen' keeps the absolute full-screen layout.)
    include_ellipse = SHOW_GAUSSIAN_ELLIPSE or ELLIPSE_ONLY
    centers = {}
    half = 0.0
    for ti, (x, y, polygon, base_centroid) in collected.items():
        c = _actual_centroid(polygon) or base_centroid
        centers[ti] = c
        if c is None:
            continue
        for px, py in polygon:
            half = max(half, abs(px - c[0]), abs(py - c[1]))
        if include_ellipse:
            st = fit_gaussian(x, y, ref_point=base_centroid)
            if st is not None:
                mx, my = st['mean']
                th = math.radians(st['angle_deg'])
                a, b = st['semi_major'], st['semi_minor']
                # Axis-aligned half-extents of the rotated ellipse, plus the
                # ellipse center's offset from the polygon centroid.
                ext_x = math.hypot(a * math.cos(th), b * math.sin(th))
                ext_y = math.hypot(a * math.sin(th), b * math.cos(th))
                half = max(half, abs(mx - c[0]) + ext_x,
                           abs(my - c[1]) + ext_y)
    half = (half or IMAGE_SIZE / 2.0) + VIEW_MARGIN_PX
    use_screen = (VIEW == 'screen')
    # One window shared by all panels, centered on the origin (the centroid).
    shared_bounds = _square_bounds(0.0, 0.0, half)

    stats_records = []
    used_axes = set()
    for r, row_params in enumerate(rows):
        for cidx, params in enumerate(row_params):
            ax = axes[r][cidx]
            used_axes.add((r, cidx))
            ti = params['trial_index']
            x, y, polygon, base_centroid = collected[ti]
            center = centers[ti]
            panel_title = (f"Trial {ti} ({fill_label(params['is_filled'])})\n"
                           f"{params['sides']} sides, rot {params['rotation']}, "
                           f"step {params['step']}")

            # Shift this panel's geometry so its centroid sits at the origin;
            # every panel then uses the same shared_bounds. Distances/angles fed
            # to the statistics plots are translation-invariant, so unaffected.
            actual_c_abs = _actual_centroid(polygon)
            if use_screen or center is None:
                bounds = None
                poly_p, x_p, y_p, base_p = polygon, x, y, base_centroid
                actual_p = actual_c_abs
            else:
                cx, cy = center
                bounds = shared_bounds
                poly_p = [(px - cx, py - cy) for px, py in polygon]
                x_p = [xi - cx for xi in x]
                y_p = [yi - cy for yi in y]
                base_p = ((base_centroid[0] - cx, base_centroid[1] - cy)
                          if base_centroid is not None else None)
                actual_p = ((actual_c_abs[0] - cx, actual_c_abs[1] - cy)
                            if actual_c_abs is not None else None)

            _n_fix, mappable, stats = _draw_trial_data(
                ax, x_p, y_p, poly_p, base_p, panel_title,
                fixation_style=fixation_style, color=color, alpha=alpha,
                show_actual=show_actual, show_base=show_base, bounds=bounds,
                legend=False, title_fontsize=9,
                show_ellipse=show_ellipse, ellipse_only=ellipse_only)
            if bounds is not None:
                ax.set_xlabel('X offset from centroid (px)', fontsize=9)
                ax.set_ylabel('Y offset from centroid (px)', fontsize=9)
            stats_records.append({
                'trial_index': ti, 'sides': params['sides'],
                'rotation': params['rotation'], 'step': params['step'],
                'is_filled': params['is_filled'],
                'base_centroid': base_p, 'actual_centroid': actual_p,
                'stats': stats,
            })
            # A small colorbar beside each panel, so it never overlaps the data.
            if mappable is not None:
                cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04,
                                    shrink=0.7)
                cbar.ax.tick_params(labelsize=6)
                cbar.set_label('Fixation density', fontsize=7)

    # Hide any unused axes in the grid (e.g. a shorter condition row).
    for r in range(nrows):
        for cidx in range(ncols):
            if (r, cidx) not in used_axes:
                axes[r][cidx].axis('off')

    fixed = (f"{sides} sides, step {step}" if group_by == 'rotations'
             else f"{sides} sides, rotation {rotation}")
    fill_title = {'filled': 'image-filled', 'unfilled': 'unfilled',
                  'both': 'filled + unfilled'}.get(fill, fill)
    fig.suptitle(f"Multi-trial ({group_by}) - {fixed}  |  {fill_title}  |  "
                 f"varying {varying}  |  {len(sessions)} sessions", fontsize=14)

    # Shared legend from the first populated panel, placed OUTSIDE the panels
    # (below the grid) so it never obscures the graphs. constrained_layout
    # reserves space for an 'outside' legend.
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='outside lower center',
                   ncol=len(labels), fontsize=9)

    # Automated statistical-inference figure for the grouped variable.
    if show_stats_plot:
        plot_multi_statistics(stats_records, group_by, fill)

    plt.show()


# --------------------------------------------------------------------------- #
# Entry point -- driven by the CONFIGURATION block at the top of the file
# --------------------------------------------------------------------------- #
def main():
    if PLOT_MODE == 'single':
        plot_single_subject(EXAMPLE_SESSION, TARGET_TRIAL)
    elif PLOT_MODE == 'aggregated':
        plot_aggregated_subjects(ROOT_DIR, TARGET_TRIAL)
    elif PLOT_MODE == 'multi':
        plot_multi_trials(ROOT_DIR)
    elif PLOT_MODE == 'analytics':
        plot_all_data_statistics(ROOT_DIR)
    else:
        print(f"Unknown PLOT_MODE={PLOT_MODE!r}. "
              f"Use 'single', 'aggregated', 'multi', or 'analytics'.")


if __name__ == '__main__':
    main()
