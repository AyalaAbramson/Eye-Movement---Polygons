# Gaze-Contingent Polygon Experiment

A Python framework for running an eye-tracking psychophysical experiment on
polygon stimuli and for analyzing the resulting gaze data. Participants view
parametrically deformed polygons (vertex stretching, concavity, rotation, image
fill) while an EyeLink 1000 Plus records their fixations; an offline analysis
tool then quantifies gaze offsets, fixation-ellipse shape/orientation, and
screen-center bias.

The codebase has two stages that share no runtime state - they communicate only
through the files each session writes to disk:

- Acquisition: `run_experiment.py`, `core_functions.py`, `eyelink_interface.py`
- Analysis: `fixation_visualization.py`

Acquisition writes, per session, a `trials.csv` and an EyeLink `.edf` (converted
to `.asc`); analysis reads those back.

---

## File overview and data flow

```
run_experiment.py        (configure + orchestrate one session)
   |  imports generate_auto_polygon, generate_manual_polygon, run_full_experiment
   v
core_functions.py        (build stimuli + run the Pygame trial loop)
   |  calls eyelink_interface for the tracker (fixation gating, messages)
   v
eyelink_interface.py     (EyeLink connect / calibrate / record / download)
   |
   v
data/raw/<participant>/<session>/  ->  trials.csv  +  edf/<name>.asc
   |
   v
fixation_visualization.py   (offline: read those files, plot + statistics)
```

---

## run_experiment.py

The master script and the only file you run for data collection. It holds the
full configuration block and drives one session end to end.

What it does:
- Defines all experiment parameters: image size, display/fixation timings, data
  paths, debug toggle, and the EyeLink settings (address, dummy mode,
  calibration type, fixation window/dwell/timeout).
- Prompts for participant info and creates the session output folder under
  `data/raw/`.
- Builds the trial list. `generate_auto_polygon` (from `core_functions`) is
  called once per condition to pre-render every stimulus. The automated set is a
  balanced factorial of vertex count, stretch step, rotation, and image fill;
  a fixed `random.seed(42)` shuffle makes the order reproducible. It also builds
  the memory-task probes shown between blocks.
- Optionally connects and calibrates the tracker through `eyelink_interface`,
  opens the per-session `trials.csv` writer, then hands everything to
  `run_full_experiment`.
- On finish or abort, downloads the `.edf` from the Host PC and closes the
  tracker.

How to use it:
1. Edit the configuration block at the top (image folders, `USE_EYELINK`,
   `EYELINK_DUMMY_MODE`, `EYELINK_ADDRESS`, `DEBUG_MODE`, timings).
2. Run `python run_experiment.py`.
3. Runtime keys: `SPACE` overrides a fixation-wait screen; `Q`/`ESC` safely
   aborts, clears the screen, and triggers the `.edf` download.

For a machine without the tracker, set `USE_EYELINK = False` or
`EYELINK_DUMMY_MODE = True` to run the visual/behavioral flow only.

---

## core_functions.py

The execution engine: stimulus geometry plus the real-time Pygame trial loop. It
is imported by `run_experiment.py` and, in turn, calls `eyelink_interface` for
the tracker.

Functions:

- `generate_manual_polygon(manual_radii, manual_angles_deg, rotation_deg, size,
  texture_path)`
  Builds a polygon directly from explicit per-vertex radius and angle lists,
  optionally rotated and filled with a texture image (otherwise solid black on a
  transparent field). Returns `(image, actual_points, None)` - there is no base
  skeleton for manual shapes. Not used in the current experiment.

- `generate_auto_polygon(num_vertices, step, rotation_deg, target_idx,
  concave_idx, concave_ratio, size, texture_path)`
  Builds a regular n-gon and applies the parametric manipulations: `step`
  stretches the `target_idx` vertex outward (with a zero-point convention that
  yields a flat edge at step 0 and a collapsed vertex at step -1), `concave_idx`
  and `concave_ratio` fold a chosen vertex inward, and `rotation_deg` rotates the
  whole shape. Fills with a texture when `texture_path` is given. Returns
  `(image, actual_points, base_points)`, where `base_points` is the undeformed
  regular skeleton (used later to recover the trial geometry during analysis).

- `run_full_experiment(trial_list, display_duration_sec, debug, tracker,
  exp_clock, screen, trial_writer, trial_file, fixation_window_px,
  fixation_required_ms, fixation_max_wait_s, block_size, memory_probes,
  block_intro_lines)`
  The main full-screen Pygame loop. For each trial it: shows the fixation cross
  and waits for a stable, gaze-contingent fixation (via
  `eyelink_interface.wait_for_fixation_gaze`, with a keypress/timeout override);
  flushes and displays the stimulus for `display_duration_sec`, sending precise
  `TRIAL_START` / `STIM_ON` / `STIM_OFF` messages to the tracker; then clears the
  screen and writes one CSV row (trial index, timing, stimulus placement, and
  the polygon points in image space). It groups trials into blocks, runs a
  memory task after each block, and writes block markers. When `debug=True` it
  overlays the grids, base skeletons, and vertex vectors instead of the clean
  stimulus.
  Nested helpers: `show_text_screen` (instruction/wait screens),
  `show_memory_task` (the between-block "did this appear?" probe), and
  `write_block_marker` (a block-start row in the CSV).

You do not run this file directly; `run_experiment.py` supplies all arguments.

---

## eyelink_interface.py

A thin, self-contained wrapper around the SR Research `pylink` API for the
EyeLink 1000 Plus. `pylink` is imported lazily, so the file can be read on
machines without the SR Research kit; the experiment itself must run on the lab
PC. All gaze coordinates use Pygame's top-left-origin pixel space.

It provides, roughly in call order: `ExpClock` (a `perf_counter`-based clock
whose timestamps are written into the tracker messages and later used to align
the `.asc` and `.csv` clocks), `connect_eyelink`, `setup_edf`,
`configure_tracker`, `open_calibration_graphics`, `do_calibration`,
`start_recording` / `stop_recording`, `wait_for_fixation_gaze` (the gaze-gated
fixation check used by the trial loop), and `close_tracker` (closes the EDF,
downloads it from the Host PC, and disconnects). A dummy mode allows an
end-to-end run without hardware. `run_experiment.py` uses it for
setup/calibration/teardown, and `core_functions.run_full_experiment` uses it for
per-trial fixation gating and messaging.

---

## fixation_visualization.py

The offline analysis and plotting tool. It reads the files produced by the
acquisition stage - no tracker or Pygame needed - and requires `numpy`,
`pandas`, `matplotlib`, `scipy`, and `shapely`.

What it does:
- Discovers each session folder (any directory containing a `trials.csv`) under
  the data root, reads that `trials.csv` and the single `edf/*.asc`, and aligns
  the two clocks using the first `TRIAL_START` message.
- Extracts the valid in-window fixations per trial (dropping the post-onset
  saccade-latency window and, optionally, fixations far outside the polygon),
  fits a 2D Gaussian to the fixation cloud, and derives metrics: offset of the
  gaze center from the stimulus centroids, ellipse axis lengths, and ellipse
  orientation.
- Produces one of four outputs selected by `PLOT_MODE`:
  - `single` - one participant, one trial (polygon + fixations + fitted ellipse).
  - `aggregated` - all participants, one trial.
  - `multi` - many related trials side by side (all rotations of a polygon, or
    all stretch steps at a fixed rotation).
  - `analytics` - dataset-wide statistical figures (axis lengths, centroid
    offsets vs step and vs rotation, orientation vs rotation, and a
    screen-center bias analysis), with optional per-subject averaging,
    significance tests, and printed p-values.

How to use it:
1. Edit the CONFIGURATION block at the top: set `ROOT_DIR` to the data root, pick
   `PLOT_MODE`, and adjust the analytics options (`ANALYTICS_FILL`,
   `ANALYTICS_SIDES/ROTATIONS/STEPS`, `USE_SUBJECT_AVERAGES`, `DISTANCE_UNITS`,
   fixation-window and boundary filters, etc.).
2. Run `python fixation_visualization.py`.

This file only reads the acquisition outputs; it never talks to the tracker.

---

## Setup

Acquisition (lab PC): Python 3.8+, `pygame`, `numpy`, `Pillow`, and the SR
Research EyeLink Developers Kit (`pylink`).

Analysis (any machine): Python 3.8+, `numpy`, `pandas`, `matplotlib`, `scipy`,
`shapely`.

Data layout written by acquisition and read by analysis:

```
data/raw/<participant_id>/<session>/
    trials.csv
    session_metadata.json
    edf/<name>.asc   (convert the .edf with SR Research's edf2asc)
```
