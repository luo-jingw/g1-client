"""Inference loop for an ARC-RESAMPLED EEF checkpoint (pi/eef-arc, config pi05_g1_eef_scube_arc).

WHAT CHANGED vs openpi/main_eef.py
----------------------------------
The robot control path, the kinematics, the prefetch overlap, the time-alignment and the cross-fade
are all main_eef's, untouched. Exactly one thing is inserted: every chunk that comes back from the
server is RE-TIMED onto the control grid before anything dispatches it.

This is not optional for an arc checkpoint. Its chunk points are spaced by CURVATURE, not by time --
consecutive points differ in distance by more than an order of magnitude within one chunk. main_eef
dispatches one row per tick, which commands a speed that tracks the point spacing: the velocity
clamp in ArmController saturates, the arm falls behind, then catches up. Re-timing converts the
(H, 16) arc chunk into an (N, 16) chunk whose rows ARE one control tick apart, after which every
downstream assumption in main_eef is true again.

Do NOT point this at a checkpoint trained with `delta_timestamps` labels -- those points are already
equally spaced in time, and re-timing them would stretch the trajectory for no reason. Use
main_eef.py for those.

HOW IT HOOKS IN
---------------
main_eef applies apply_raisez() to every chunk the moment it arrives -- the first one inline and
every prefetched one -- and nothing else touches the array before dispatch. That makes it the single
seam where re-timing can be inserted without forking the 120-line scheduling loop, so this module
wraps that function rather than copying the loop. If main_eef's loop grows another place that
receives a chunk, this wrapper must be revisited.

--exec-steps NOW MEANS SOMETHING DIFFERENT
------------------------------------------
Steps are control ticks, and a re-timed chunk is much longer than the raw horizon: 32 arc points
covering ~300 recorded frames become ~9 s, i.e. ~135 ticks at 15 Hz. Leaving --exec-steps at 0
(full horizon) therefore runs ~9 s open loop between queries. Set it to 2-3 s worth of ticks
(30-45 at 15 Hz) unless you specifically want the long open-loop.

CALIBRATION (measured on stack-cube-eef, 298 chunks, recorded span p50 9.78 s)
------------------------------------------------------------------------------
Playback duration as a fraction of the duration the same stretch took in the recording:

    v_nominal  dt_min   ratio        v_nominal  dt_min   ratio
      0.050     0.04    1.066          0.065     0.04    0.823
      0.055     0.04    0.969  <-      0.070     0.04    0.764
      0.060     0.04    0.891          0.075     0.04    0.717

Default is 0.055 / 0.04: the demonstration's own pace, which is the conservative choice for a first
run on hardware. 0.060 gives 0.89x if you want it brisker. The dt_min floor binds on ~14% of steps
-- those are the dense grasp stretches, and the floor is what stops playback rushing them.

Usage (run from the repo root, same as main_eef.py plus the arc flags):
  python openpi/main_eef_arc.py \\
      --iface enp0s31f6 --server-host 1.2.3.4 --server-port 8000 \\
      --prompt "Stack the blocks by color: put the red block in the center, then stack the blue block on the red block, then stack the yellow block on the blue block." \\
      --exec-steps 40
"""

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))       # openpi/ -> import main_eef
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import arc_playback
import main_eef

log = logging.getLogger("g1_openpi.main_eef_arc")

# Stash the originals ON the target module, so re-importing or reloading this one picks up the
# genuine functions rather than a previous patch. Without the guard a second install would wrap the
# wrapper and recurse until the stack blows.
_orig_apply_raisez = getattr(main_eef, "_arc_orig_apply_raisez", main_eef.apply_raisez)
_orig_run = getattr(main_eef, "_arc_orig_run", main_eef.run)
main_eef._arc_orig_apply_raisez = _orig_apply_raisez
main_eef._arc_orig_run = _orig_run


def _make_retimer(arc, args):
    """Wrap main_eef.apply_raisez so every arriving chunk is re-timed first."""
    state = {"n": 0}

    def retimed(actions: np.ndarray, raisez_mm: float) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float64)
        if arc.arc_off:
            return _orig_apply_raisez(actions, raisez_mm)

        report = arc_playback.schedule_report(
            actions, args.control_hz, arc.arc_v_nominal, arc.arc_dt_min)
        out = arc_playback.retime(
            actions, control_hz=args.control_hz, v_nominal=arc.arc_v_nominal,
            dt_min=arc.arc_dt_min, max_seconds=arc.arc_max_seconds, interp=arc.arc_interp)
        state["n"] += 1
        log.info(
            f"[arc {state['n']}] {report['points']} points -> {len(out)} ticks "
            f"({report['seconds']:.2f}s @ {args.control_hz:g} Hz) | "
            f"step mm p05/p50/p95 {report['dist_mm_p05_p50_p95']} | "
            f"dt ms {report['dt_ms_p05_p50_p95']} | at dt_min {report['at_dt_min_pct']}%")
        # Raise-Z after re-timing: it is a constant offset, so the order does not matter
        # numerically, but doing it second keeps the logged ranges describing what is dispatched.
        return _orig_apply_raisez(out, raisez_mm)

    return retimed


def _run(args):
    arc = args._arc
    if arc.arc_off:
        log.warning("--arc-off: chunks dispatched one row per tick, as main_eef does. Correct ONLY "
                    "for an equal-time checkpoint; an arc checkpoint will judder.")
    else:
        log.info(f"arc re-timing ON: v_nominal={arc.arc_v_nominal} m/s, dt_min={arc.arc_dt_min}s, "
                 f"interp={arc.arc_interp}, control_hz={args.control_hz:g}")
        if args.exec_steps <= 0:
            log.warning("--exec-steps 0 with re-timing means running a whole re-timed chunk "
                        "(~9 s) open loop before re-querying. Set 30-45 for 2-3 s at 15 Hz.")
    main_eef.apply_raisez = _make_retimer(arc, args)
    return _orig_run(args)


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--arc-v-nominal", type=float, default=0.055,
                    help="Nominal end-effector speed (m/s) used to give each chunk point a "
                         "duration. Lower = slower playback. Default 0.055 reproduces the "
                         "demonstration pace (0.97x); see the module docstring for the table.")
    ap.add_argument("--arc-dt-min", type=float, default=0.04,
                    help="Floor on a segment's duration (s). Not a safety margin: points inside a "
                         "grasp are dense in space but represent real elapsed time, and a purely "
                         "distance-proportional schedule would rush them (default 0.04).")
    ap.add_argument("--arc-max-seconds", type=float, default=0.0,
                    help="Backstop cap on the wall time one chunk may cover (0 = off). The normal "
                         "way to bound open-loop time is --exec-steps.")
    ap.add_argument("--arc-interp", choices=("pchip", "linear"), default="pchip",
                    help="Interpolation between chunk points. Measured level on this dataset at "
                         "15-120 Hz (linear has the slightly lower peak speed); pchip is the "
                         "default because it matches how the labels were built.")
    ap.add_argument("--arc-off", action="store_true",
                    help="Disable re-timing and dispatch one row per tick, i.e. behave exactly like "
                         "main_eef.py. For A/B against the baseline checkpoint.")
    arc, rest = ap.parse_known_args()

    # main_eef.main() builds its own parser and then calls main_eef.run; patch run so the arc
    # settings arrive with the parsed args (control_hz and exec_steps are needed there).
    main_eef.run = lambda args: _run(_attach(args, arc))
    sys.argv = [sys.argv[0], *rest]
    main_eef.main()


def _attach(args, arc):
    args._arc = arc
    return args


if __name__ == "__main__":
    main()
