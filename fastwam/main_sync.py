"""Synchronous (no-prefetch) EEF inference loop for the G1 against a FastWAM policy
server (FastWAM/serve.py) — the simplest of the three fastwam/ EEF loops.

WHAT THIS IS
------------
One chunk at a time, strictly sequential:

    obs -> infer (BLOCKING, arm holds its last pose) -> execute ALL H actions -> obs -> ...

No prefetch thread, no time-alignment, no receding horizon, no weighted merge —
every action the model returns is dispatched, in order, before the next
observation is taken.

THE THREE fastwam/ EEF LOOPS
----------------------------
    script         schedule                          motion        obs staleness
    -----------    -------------------------------   -----------   -------------------
    main_sync.py   infer -> execute all H            stop-and-go   fresh (0 steps)
    main_eef.py    prefetch `lead` steps early,      continuous    lead steps stale
                   swap chunks, cross-fade N steps
    main_mask.py   merge every lead+wait steps,      continuous    wait steps stale
                   weighted joint-space blend

WHY YOU'D WANT THIS
-------------------
* The observation is exactly the state the chunk was predicted from — nothing is
  dropped for time-alignment and no stale action ever reaches the arm. That makes
  it the reference for "is the checkpoint any good?" before tuning a schedule.
* Nothing is blended, so what you watch on the robot is the model's raw chunk.
* No threads: FK, IK and gravity_torque all run on the main thread, so ONE
  G1DualArmKinematics is enough and there is no pinocchio-Data hazard at all.

THE COST
--------
The arm stands still for the whole inference (FastWAM's diffusion sampler, see the
server's `infer ok: ... in NNNms` line) between chunks, so motion is stop-and-go:
at 30 Hz a 32-step chunk is ~1.07 s of movement, and a ~700 ms sampler means the
arm is moving only ~60% of the time. The end-of-run summary prints that duty
cycle. Contact-rich phases (closing on an object) are where the pauses hurt most —
if that's a problem, use main_eef.py or main_mask.py instead.

Because there is no worker thread, the per-step EEF->joint IK is on the control
loop's critical path. The summary counts steps that missed their dt deadline; if
that number is large, lower --control-hz.

Gravity feedforward keeps holding the last commanded pose during the inference
pause (the arm/gripper publish threads keep running), so the arm doesn't sag
between chunks and the next observation stays in-distribution.

Press 'r' at any time to ramp back to the ready pose and re-arm; the grippers open
first so a held object is dropped in place. Keys are polled between steps, so a
press during the blocking infer is picked up as soon as the chunk starts.

Precondition: robot already in 'ai' motion mode (set via the Unitree app), and
FastWAM/serve.py is serving an EEF-trained checkpoint.

Usage (run from the repo root):
  python fastwam/main_sync.py \\
      --iface enp0s31f6 \\
      --server-host 1.2.3.4 \\
      --server-port 8000 \\
      --prompt "pick the red bottle"
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)                        # repo root -> import g1_client
# APPEND (not insert) the openpi folder: it is only needed for eef_kinematics, and
# it also holds a main_eef.py. Putting it ahead of this script's own directory
# would shadow fastwam/main_eef.py below and silently send openpi's LeRobot-keyed
# observation ("observation.images.*"/"observation.state") to serve.py, which
# expects {"image", "state", "prompt"} and fails with KeyError: 'image'.
sys.path.append(os.path.join(_ROOT, "openpi"))   # -> eef_kinematics

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from g1_client.arm_controller import ArmController
from g1_client.gripper_controller import GripperController, GRIPPER_MIN, GRIPPER_MAX
from g1_client.camera_client import CameraClient
from fastwam_policy import FastWAMPolicy

from eef_kinematics import G1DualArmKinematics, DEFAULT_URDF, DEFAULT_ASSETS

# Reuse everything that is identical to the EEF FastWAM client — controllers,
# EEF<->obs assembly, EEF channel layout, the keyboard poller and the
# release/init/cleanup/standby sequence are untouched; only the (much simpler)
# schedule below is different.
from main_eef import (
    LEFT_EEF_CHANNELS, RIGHT_EEF_CHANNELS,
    LEFT_GRIPPER_CHANNEL, RIGHT_GRIPPER_CHANNEL, IK_WARN_M,
    build_obs, apply_raisez, log_chunk_ranges,
    _pct, _stat, _timing_rec, _KeyPoller,
    _release_grippers, _initialize_pose, _wait_for_operator, _cleanup,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("g1_fastwam.sync")


# ---------- latency profiling ----------

def _summarize_timing(infer_recs, chunk_recs, args) -> None:
    """End-of-run summary: per-infer latency, the stop-and-go duty cycle, IK cost
    and any missed control deadlines.

    The duty cycle is what this loop trades away for a fresh observation: with no
    prefetch, wall time is execute + infer + execute + infer..., so
    duty = sum(execute) / (sum(execute) + sum(infer)) is the fraction of the run
    the arm was actually moving. wait_recv is the blocking recv = network +
    FastWAM's sampler; the sampler dominates, so compare it with the server's
    `infer ok: ... in NNNms` log line.
    """
    if not infer_recs:
        return
    log.info("=" * 64)
    log.info(f"per-infer latency over {len(infer_recs)} calls (ms):")
    log.info(f"  {'component':<16} {'min':>7} {'p50':>7} {'p95':>7} {'max':>7} {'mean':>7}")
    for label, key in (("pack", "pack_ms"), ("send", "send_ms"),
                       ("wait_recv(gpu)", "wait_recv_ms"), ("unpack", "unpack_ms"),
                       ("wall(total)", "wall_ms")):
        mn, p50, p95, mx, me = _stat([r[key] for r in infer_recs])
        log.info(f"  {label:<16} {mn:7.1f} {p50:7.1f} {p95:7.1f} {mx:7.1f} {me:7.1f}")
    up = [r["bytes_sent"] for r in infer_recs if r["bytes_sent"] > 0]
    if up:
        log.info(f"  upload payload ~= {np.mean(up)/1024:.0f} KiB/infer "
                 f"(3 decoded RGB views — the FastWAM obs format; the server resizes)")
    if chunk_recs:
        ex = [r["exec_s"] for r in chunk_recs]
        inf = [r["infer_s"] for r in chunk_recs]
        steps = [r["steps"] for r in chunk_recs]
        ikm = [r["ik_max_m"] * 1e3 for r in chunk_recs]
        late = sum(r["late_steps"] for r in chunk_recs)
        total = sum(ex) + sum(inf)
        duty = sum(ex) / total if total > 0 else float("nan")
        log.info(f"executed {sum(steps)} actions over {len(chunk_recs)} chunk(s), "
                 f"{np.mean(steps):.1f} steps/chunk (nothing dropped, nothing blended away)")
        log.info(f"execute/chunk: mean={np.mean(ex):.2f}s | infer/chunk: mean={np.mean(inf):.2f}s "
                 f"p95={_pct(inf,95):.2f}s max={max(inf):.2f}s")
        log.info(f"DUTY CYCLE: arm moving {duty*100:.0f}% of the {total:.1f}s run — "
                 f"it paused {np.mean(inf):.2f}s between chunks waiting on inference")
        log.info(f"IK residual/chunk (mm): mean={np.mean(ikm):.2f} p95={_pct(ikm,95):.2f} "
                 f"max={max(ikm):.2f}")
        if late:
            log.warning(f"{late}/{sum(steps)} step(s) missed the "
                        f"{1e3/args.control_hz:.0f}ms control deadline — the per-step IK "
                        f"does not fit the dispatch budget on this machine; lower "
                        f"--control-hz so the chunk is not played back slower than "
                        f"requested.")
        else:
            log.info(f"all steps met the {1e3/args.control_hz:.0f}ms control deadline.")
        if duty < 0.6:
            log.warning(f"duty cycle {duty*100:.0f}%: the stop-and-go pauses dominate. That "
                        f"is inherent to this loop — cut the server's --num-inference-steps, "
                        f"or switch to fastwam/main_eef.py (prefetch) / main_mask.py "
                        f"(weighted merge) for continuous motion.")
    log.info("=" * 64)


# ---------- inference loop ----------

def _run_sync_loop(arm, grip, cam, policy, kin, args) -> None:
    """infer -> execute the whole chunk -> infer -> ... with nothing overlapped.

    Each chunk is dispatched from index 0 (the observation was taken immediately
    before the infer, so index 0 is already wall-clock aligned — this is the one
    schedule that needs no time-alignment). Each EEF row is IK-solved to 14 joint
    targets, warm-started from the previous solution so the redundant elbow DOF
    stays on one branch for the whole run.

    While policy.infer() blocks, the arm/gripper publish threads keep holding the
    last commanded target (with gravity feedforward if --tauff-scale > 0), so the
    inter-chunk pause is a stationary hold, not a sag.
    """
    dt = 1.0 / args.control_hz
    prompt = args.prompt
    if args.tauff_scale > 0:
        log.info(f"gravity feedforward ON (--tauff-scale {args.tauff_scale}): the arm holds "
                 f"commanded poses against gravity, including during the inference pause")
    else:
        log.warning("gravity feedforward OFF (--tauff-scale 0): the arm will sag below "
                    "commanded poses — state feedback drifts out of the training distribution")

    infer_recs = []   # one per inference: latency breakdown
    chunk_recs = []   # one per chunk: infer/execute seconds, steps, IK residual, late steps
    with _KeyPoller() as keys:
        log.info("Press [r] at any time to reset to the ready pose and re-arm.")
        while True:
            # IK warm start carried across chunks, for elbow-branch continuity.
            ik_q = np.asarray(arm.get_arm_q(), dtype=np.float64)
            last_cmd = None       # last commanded 16-vec (JOINT space), for the ramp-in
            reset_requested = False

            for c in range(1, args.max_chunks + 1):
                # ---- infer (blocking; the arm holds its last commanded pose) ----
                if c == 1:
                    log.info(f"First inference (prompt={prompt!r}) — FastWAM warm-up may be slow")
                infer_t0 = time.time()
                result = policy.infer(build_obs(cam, arm, grip, kin, prompt))
                infer_s = time.time() - infer_t0
                actions = np.asarray(result["actions"], dtype=np.float64)
                if actions.ndim != 2 or actions.shape[1] < 16:
                    raise RuntimeError(f"Unexpected action shape {actions.shape} (want [H, 16])")
                if c == 1 and args.raisez:
                    log.info(f"--raisez {args.raisez:.1f} mm: offsetting every EEF Z target by "
                             f"{args.raisez*1e-3:+.4f} m (pelvis frame)")
                actions = apply_raisez(actions, args.raisez)
                rec = _timing_rec(dict(policy.last_timing or {}))
                infer_recs.append(rec)
                log_chunk_ranges(c - 1, actions)

                # ---- execute EVERY action in the chunk, in order ----
                H = actions.shape[0]
                ik_max_m = 0.0
                late = 0
                exec_t0 = time.time()
                for i in range(H):
                    if arm.faulted():
                        raise RuntimeError("ArmController control thread faulted — aborting")
                    if keys.poll() == "r":
                        reset_requested = True
                        break
                    tic = time.time()

                    a_eef = actions[i]
                    ik_q, pos_err = kin.solve_ik(a_eef[LEFT_EEF_CHANNELS],
                                                 a_eef[RIGHT_EEF_CHANNELS], ik_q)
                    ik_max_m = max(ik_max_m, pos_err)
                    if pos_err > IK_WARN_M:
                        log.warning(f"[chunk {c} step {i}] IK residual {pos_err*1e3:.1f} mm — "
                                    f"model predicted a barely-reachable EEF pose")
                    a = np.concatenate([ik_q, a_eef[[LEFT_GRIPPER_CHANNEL, RIGHT_GRIPPER_CHANNEL]]])
                    # Ramp the first --blend-steps in from the last commanded pose. The
                    # obs is fresh here, so the jump is usually small — but the arm has
                    # been standing still through the inference pause, and a chunk that
                    # opens with a big step would otherwise be a lurch out of the hold.
                    if last_cmd is not None and i < args.blend_steps:
                        alpha = (i + 1) / (args.blend_steps + 1)
                        a = (1.0 - alpha) * last_cmd + alpha * a
                    arm.set_arm_target(a[:14])
                    # Gravity feedforward: hold the commanded pose instead of sagging
                    # under kp — matches the collection-time dynamics (tau=sol_tauff) the
                    # policy was trained on, so state feedback stays in-distribution.
                    if args.tauff_scale > 0:
                        arm.set_arm_tauff(kin.gravity_torque(a[:14], args.tauff_scale))
                    grip.set_targets(
                        float(np.clip(a[LEFT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
                        float(np.clip(a[RIGHT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
                    )
                    last_cmd = a

                    sleep = dt - (time.time() - tic)
                    if sleep > 0:
                        time.sleep(sleep)
                    else:
                        late += 1   # IK + dispatch overran dt: this chunk plays back slow
                exec_s = time.time() - exec_t0

                if reset_requested:
                    break

                chunk_recs.append({"infer_s": infer_s, "exec_s": exec_s, "steps": H,
                                   "ik_max_m": ik_max_m, "late_steps": late})
                log.info(f"[chunk {c}] infer={infer_s*1e3:.0f}ms (gpu={rec['wait_recv_ms']:.0f}ms) "
                         f"-> executed {H}/{H} steps in {exec_s:.2f}s "
                         f"ik_max={ik_max_m*1e3:.2f}mm"
                         + (f" late={late}" if late else "")
                         + f" | arm idle {infer_s/(infer_s+exec_s)*100:.0f}% of this cycle")

            if not reset_requested:
                break  # ran to --max-chunks — done

            # 'r' pressed: release, drop feedforward, ramp back to the ready pose,
            # and wait for Enter before starting a fresh session from the top.
            log.info("[r] reset requested — returning to ready pose")
            # Let go first, while the arm is still holding its pose under
            # feedforward, so the object is dropped where it is instead of being
            # carried back to the ready pose.
            _release_grippers(grip, args)
            if args.tauff_scale > 0:
                arm.set_arm_tauff(np.zeros(14))
            _initialize_pose(arm, grip, args)
            log.info("Press [Enter] to resume inference, or [Ctrl+C] to abort.")
            keys.wait_enter()

    # Drop the feedforward before run() ramps back to the ready pose, so that
    # move runs with the arm's default (tau=0) dynamics.
    if args.tauff_scale > 0:
        arm.set_arm_tauff(np.zeros(14))
    _summarize_timing(infer_recs, chunk_recs, args)


# ---------- entry point ----------

def run(args) -> None:
    # EEF mode needs the kinematics for build_obs (measured q -> EEF state), the
    # per-step IK and the gravity feedforward. ONE instance is enough here: this
    # loop has no worker thread at all, so nothing can touch pinocchio's Data
    # concurrently. A bad URDF fails here, before DDS.
    log.info(f"Loading G1 dual-arm model from {args.urdf}")
    kin = G1DualArmKinematics(args.urdf, args.assets)

    log.info(f"Initializing DDS on {args.iface}")
    ChannelFactoryInitialize(0, args.iface)

    arm = ArmController(publish_hz=50.0, velocity_limit=args.velocity_limit)
    arm.start()
    grip = None
    cam = None
    policy = None
    try:
        grip = GripperController(publish_hz=200.0)
        grip.start()
        cam = CameraClient(host=args.image_server)
        _initialize_pose(arm, grip, args)
        _wait_for_operator(args)
        log.info(f"Switching arm kp to inference value: {args.inference_kp_arm}")
        arm.set_arm_kp(args.inference_kp_arm)
        # Connect after Enter (PolicyClient waits for the server if it isn't up yet,
        # holding the arms at INIT_POSE_READY meanwhile — same as fastwam/main_eef.py).
        policy = FastWAMPolicy(host=args.server_host, port=args.server_port)
        _run_sync_loop(arm, grip, cam, policy, kin, args)
        _initialize_pose(arm, grip, args)
    finally:
        _cleanup(arm, grip, cam, policy)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iface", required=True, help="Network interface to robot, e.g. enp0s31f6")
    p.add_argument("--server-host", required=True, help="FastWAM/serve.py host or IP")
    p.add_argument("--server-port", type=int, default=8000, help="FastWAM server port (default 8000)")
    # ---- Robot I/O ----
    p.add_argument("--image-server", default="192.168.123.164",
                   help="G1 PC2 image-server host (default 192.168.123.164)")
    p.add_argument("--prompt", default="pick the red bottle")
    p.add_argument("--urdf", default=DEFAULT_URDF,
                   help="G1 URDF for FK/IK (must match the dataset conversion model)")
    p.add_argument("--assets", default=DEFAULT_ASSETS,
                   help="Directory with the URDF's mesh assets")
    p.add_argument("--raisez", type=float, default=0.0,
                   help="Raise every returned EEF Z target by this many mm (pelvis "
                        "frame, both arms) before IK. Positive = higher. Default 0 (off).")
    p.add_argument("--max-chunks", type=int, default=30,
                   help="How many chunks to infer+execute before stopping")
    p.add_argument("--control-hz", type=float, default=30.0,
                   help="Per-step dispatch rate; match your LeRobot recording fps (30). "
                        "Lower it if the summary reports missed control deadlines (the "
                        "per-step IK runs on this loop, there is no worker thread).")
    p.add_argument("--blend-steps", type=int, default=5,
                   help="Ramp the first N steps of each chunk in from the last commanded "
                        "pose, so leaving the inference hold is smooth (0 = dispatch the "
                        "model's actions completely unmodified)")
    # ---- Gravity feedforward (reduced-arm model from openpi/eef_kinematics.py) ----
    p.add_argument("--tauff-scale", type=float, default=1.0,
                   help="Scale on the gravity-compensation feedforward torque fed to the "
                        "arm each step (default 1.0 = full comp, matching how the data was "
                        "collected). Use <1 (e.g. 0.5) for a cautious first pass, or 0 to "
                        "disable (the arm then sags and state feedback drifts OOD).")
    # ---- Safety / motion limits (same as fastwam/main_eef.py) ----
    p.add_argument("--velocity-limit", type=float, default=8.0,
                   help="rad/s velocity cap on the per-tick motion clamp (default 8.0)")
    p.add_argument("--inference-kp-arm", type=float, default=80.0,
                   help="kp for shoulder/elbow once inference starts (default 80)")
    p.add_argument("--init-duration", type=float, default=2.0)
    p.add_argument("--gripper-init-duration", type=float, default=1.0)
    p.add_argument("--settle-duration", type=float, default=1.0)
    p.add_argument("--init-gripper-left", type=float, default=5.0)
    p.add_argument("--init-gripper-right", type=float, default=5.0)
    p.add_argument("--no-reset-open-gripper", action="store_false", dest="reset_open_gripper",
                   help="Do NOT open the grippers when [r] is pressed. By default a reset "
                        "releases first so a held object is dropped in place rather than "
                        "dragged back to the ready pose.")
    p.add_argument("--reset-gripper-open", type=float, default=GRIPPER_MAX,
                   help=f"Gripper target used by the [r] release, rad (default "
                        f"{GRIPPER_MAX} = fully open)")
    p.add_argument("--reset-gripper-duration", type=float, default=0.5,
                   help="Seconds to ramp the grippers open on a [r] reset (default 0.5)")
    p.add_argument("--auto-start", action="store_true",
                   help="Skip the post-init Enter prompt and start immediately.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
