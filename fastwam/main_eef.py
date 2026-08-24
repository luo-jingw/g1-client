"""EEF-space inference loop for the G1 driving a FastWAM policy server
(FastWAM/serve_fastwam_g1.py) — the EEF counterpart of fastwam/main.py.

WHAT CHANGED vs fastwam/main.py
-------------------------------
Same FastWAM transport (FastWAMPolicy), same obs wire format ({"image":[3 RGB],
"state":(16,), "prompt"}), same robot init / standby / kp-switch / cleanup, same
gravity feedforward and 'r'-reset. The ONLY change is the observation/action
space, mirroring openpi/main_eef.py:

  * obs "state" is the two EEF poses (measured arm q -> FK) + 2 grippers, not the
    14 raw joint angles.
  * the returned action [H, 16] is EEF-space:
        [:, 0:7]  left EEF pose  (x,y,z,qx,qy,qz,qw), pelvis frame
        [:, 7:14] right EEF pose (same layout)
        [:, 14]   left gripper  (rad, [GRIPPER_MIN, GRIPPER_MAX])
        [:, 15]   right gripper (rad, [GRIPPER_MIN, GRIPPER_MAX])
    Each step is IK-solved to 14 joint targets (warm-started from the previous
    solution), and the cross-fade blend happens on the IK output in JOINT space —
    blending quaternions linearly across a chunk swap would cut corners through
    SO(3).

Only ONE G1DualArmKinematics is needed: IK and FK and gravity_torque all run on
the main thread (the prefetch worker only does policy.infer, no kinematics), so
there is no pinocchio-Data thread-safety concern here.

Precondition: robot already in 'ai' motion mode (set via the Unitree app), and
FastWAM/serve_fastwam_g1.py is serving an EEF-trained checkpoint.

Usage (run from the repo root):
  python fastwam/main_eef.py \\
      --iface enp0s31f6 \\
      --server-host 1.2.3.4 \\
      --server-port 8000 \\
      --prompt "pick the red bottle"
"""

import argparse
import logging
import os
import select
import sys
import termios
import threading
import time
import tty

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import g1_client
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "openpi"))  # -> eef_kinematics

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from g1_client.arm_controller import ArmController, INIT_POSE_READY
from g1_client.gripper_controller import GripperController, GRIPPER_MIN, GRIPPER_MAX
from g1_client.camera_client import CameraClient
from fastwam_policy import FastWAMPolicy
from eef_kinematics import G1DualArmKinematics, DEFAULT_URDF, DEFAULT_ASSETS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("g1_fastwam.eef")

# Action tensor layout for the G1 EEF FastWAM checkpoint: [H, 16].
LEFT_EEF_CHANNELS = slice(0, 7)
RIGHT_EEF_CHANNELS = slice(7, 14)
LEFT_GRIPPER_CHANNEL = 14
RIGHT_GRIPPER_CHANNEL = 15

# IK residual (m) above which a dispatched step is loudly logged.
IK_WARN_M = 0.02

# The 3-camera checkpoint conditions on [cam_left_high, cam_left_wrist,
# cam_right_wrist]. ORDER MUST MATCH the training shape_meta.images order.
OBS_CAM_KEYS = [
    "observation.images.cam_left_high",    # ego / head (top of the concat)
    "observation.images.cam_left_wrist",   # left wrist
    "observation.images.cam_right_wrist",  # right wrist
]


# ---------- observation assembly ----------

def _jpeg_to_rgb(jpeg_bytes: bytes) -> np.ndarray:
    """Decode the camera client's JPEG bytes back to an RGB uint8 array.

    camera_client.get_obs_images() returns BGR-encoded JPEG. FastWAM is trained on
    LeRobot RGB frames, so we decode + BGR->RGB here to match.
    """
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode failed on a camera frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_obs(cam: CameraClient, arm: ArmController, grip: GripperController,
              kin: G1DualArmKinematics, prompt: str) -> dict:
    """Assemble one FastWAM observation with the state in EEF space.

    Same wire format as fastwam/main.py's build_obs; the only difference is that
    the measured 14 arm joints go through FK so the 16-dim state carries the two
    EEF poses the EEF checkpoint was trained on.
    """
    imgs = cam.get_obs_images()  # dict of JPEG bytes (BGR, q90), LeRobot keys
    views = [_jpeg_to_rgb(imgs[k]) for k in OBS_CAM_KEYS]
    left_q, right_q = grip.get_state()
    left_eef, right_eef = kin.fk(arm.get_arm_q())
    state = np.concatenate([left_eef, right_eef, [left_q, right_q]]).astype(np.float32)  # (16,)
    return {"image": views, "state": state, "prompt": prompt}


def apply_raisez(actions: np.ndarray, raisez_mm: float) -> np.ndarray:
    """Raise both EEF Z targets by raisez_mm (mm, pelvis frame) before IK."""
    if raisez_mm:
        actions = actions.copy()
        dz = raisez_mm * 1e-3
        actions[:, LEFT_EEF_CHANNELS.start + 2] += dz
        actions[:, RIGHT_EEF_CHANNELS.start + 2] += dz
    return actions


def log_chunk_ranges(chunk_id: int, actions: np.ndarray) -> None:
    """One-line per-arm EEF range sanity print before a chunk streams to the arm."""
    gl = actions[:, LEFT_GRIPPER_CHANNEL]
    gr = actions[:, RIGHT_GRIPPER_CHANNEL]
    log.info(f"[chunk {chunk_id}] H={actions.shape[0]} EEF ranges (pelvis frame, m):")
    for name, sl in (("L", LEFT_EEF_CHANNELS), ("R", RIGHT_EEF_CHANNELS)):
        pos = actions[:, sl][:, :3]
        quat = actions[:, sl][:, 3:7]
        qn = np.linalg.norm(quat, axis=1)
        log.info(f"    {name} x:[{pos[:,0].min():+.3f},{pos[:,0].max():+.3f}] "
                 f"y:[{pos[:,1].min():+.3f},{pos[:,1].max():+.3f}] "
                 f"z:[{pos[:,2].min():+.3f},{pos[:,2].max():+.3f}] "
                 f"|q|:[{qn.min():.3f},{qn.max():.3f}]")
    log.info(f"[chunk {chunk_id}] gripper L:[{gl.min():.2f},{gl.max():.2f}] "
             f"R:[{gr.min():.2f},{gr.max():.2f}]")


# ---------- inference loop (the "send" side) ----------

def _pct(xs, p):
    return float(np.percentile(xs, p)) if xs else float("nan")


def _stat(xs):
    """(min, p50, p95, max, mean) of a list of numbers."""
    return (min(xs), _pct(xs, 50), _pct(xs, 95), max(xs), float(np.mean(xs)))


def _timing_rec(tm):
    """Turn FastWAMPolicy.last_timing into a uniform per-infer record (ms)."""
    return {
        "wall_ms": tm.get("total_s", 0.0) * 1e3,
        "pack_ms": tm.get("pack_s", 0.0) * 1e3,
        "send_ms": tm.get("send_s", 0.0) * 1e3,
        "wait_recv_ms": tm.get("wait_recv_s", 0.0) * 1e3,
        "unpack_ms": tm.get("unpack_s", 0.0) * 1e3,
        "bytes_sent": tm.get("bytes_sent", -1),
        "bytes_recv": tm.get("bytes_recv", -1),
    }


def _summarize_timing(infer_recs, chunk_recs, args):
    """Compact end-of-run latency summary (per-infer breakdown + execute/stall).

    wait_recv is the blocking recv: network round-trip + the server's diffusion
    sampler combined. For FastWAM the sampler dominates, so this is mostly GPU time.
    """
    if not infer_recs:
        return
    budget_ms = (args.prefetch_lead / args.control_hz) * 1e3
    log.info("=" * 60)
    log.info(f"per-infer latency over {len(infer_recs)} calls (ms):")
    comps = [("wall(total)", "wall_ms"), ("wait_recv(gpu)", "wait_recv_ms")]
    for label, key in comps:
        mn, p50, p95, mx, me = _stat([r[key] for r in infer_recs])
        log.info(f"  {label:<16} {mn:7.1f} {p50:7.1f} {p95:7.1f} {mx:7.1f} {me:7.1f}")
    if chunk_recs:
        ex = [r["exec_s"] for r in chunk_recs]
        jw = [r["join_wait_s"] * 1e3 for r in chunk_recs]
        stalled = sum(1 for x in jw if x > 1.0)
        log.info(f"execute/chunk: mean={np.mean(ex):.2f}s | join_wait: mean={np.mean(jw):.0f}ms "
                 f"p95={_pct(jw,95):.0f}ms | stalled {stalled}/{len(jw)} chunks "
                 f"(infer didn't finish within the {budget_ms:.0f}ms prefetch budget)")
        ik = [r["ik_max_m"] * 1e3 for r in chunk_recs if "ik_max_m" in r]
        if ik:
            log.info(f"IK residual/chunk (mm): mean={np.mean(ik):.2f} p95={_pct(ik,95):.2f} "
                     f"max={max(ik):.2f}")
        if stalled:
            log.warning(f"{stalled} chunk(s) STALLED at the boundary — FastWAM inference "
                        f"is slower than the overlap window. Lower --control-hz, raise "
                        f"--prefetch-lead, or cut --num-inference-steps.")
    log.info("=" * 60)


class _KeyPoller:
    """Non-blocking single-key reader on a POSIX terminal.

    Puts stdin in cbreak mode so keypresses arrive without Enter. `poll()` returns
    the pending key char (or None) without blocking; `wait_enter()` blocks until
    Enter. Restores terminal settings on exit. If stdin is not a tty (e.g. piped),
    it degrades to a no-op poll and a plain input() wait.
    """

    def __init__(self):
        self._tty = sys.stdin.isatty()
        self._fd = sys.stdin.fileno() if self._tty else None
        self._old = None

    def __enter__(self):
        if self._tty:
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if self._tty and self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self):
        if not self._tty:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def wait_enter(self):
        if not self._tty:
            try:
                input("")
            except EOFError:
                pass
            return
        while sys.stdin.read(1) not in ("\n", "\r"):
            pass


def _infer_worker(policy: FastWAMPolicy, obs: dict, box: dict) -> None:
    """Run one blocking infer on a daemon thread; stash result/exception/timing.

    Returns the raw EEF chunk — IK happens per-step on the main thread, so this
    worker never touches the kinematics (no pinocchio-Data contention).
    """
    try:
        result = policy.infer(obs)
        box["timing"] = dict(policy.last_timing or {})
        actions = np.asarray(result["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] < 16:
            raise RuntimeError(f"Unexpected action shape {actions.shape} (want [H, 16])")
        box["actions"] = actions
    except BaseException as e:  # surface to the main thread
        box["err"] = e


def _run_inference_loop(arm, grip, cam, policy, kin, args) -> None:
    """Receding-horizon loop with one-chunk prefetch + boundary smoothing, EEF.

    Same schedule as fastwam/main.py (prefetch overlap, time-alignment, cross-fade)
    plus a per-tick EEF->joint IK step: each action row is IK-solved to 14 joint
    targets (warm-started from the previous solution), and the cross-fade blend is
    done on the IK output in JOINT space. Press 'r' to reset to the ready pose.
    """
    dt = 1.0 / args.control_hz
    prompt = args.prompt
    if args.tauff_scale > 0:
        log.info(f"gravity feedforward ON (--tauff-scale {args.tauff_scale}): the arm "
                 f"holds commanded poses against gravity, matching collection dynamics")
    else:
        log.warning("gravity feedforward OFF (--tauff-scale 0): the arm will sag below "
                    "commanded poses — state feedback drifts out of the training distribution")

    infer_recs = []
    chunk_recs = []

    # Outer loop so pressing 'r' mid-run can ramp back to the ready pose, wait for
    # Enter, and start a fresh inference session from the top.
    with _KeyPoller() as keys:
        log.info("Press [r] at any time to reset to the ready pose and re-arm.")
        while True:
            # First chunk is a blocking infer (nothing to overlap it against yet).
            log.info(f"First inference (prompt={prompt!r}) — FastWAM warm-up may be slow")
            result = policy.infer(build_obs(cam, arm, grip, kin, prompt))
            actions = np.asarray(result["actions"], dtype=np.float64)
            if actions.ndim != 2 or actions.shape[1] < 16:
                raise RuntimeError(f"Unexpected action shape {actions.shape} (want [H, 16])")
            if args.raisez:
                log.info(f"--raisez {args.raisez:.1f} mm: offsetting every EEF Z target by "
                         f"{args.raisez*1e-3:+.4f} m (pelvis frame)")
            actions = apply_raisez(actions, args.raisez)
            log_chunk_ranges(0, actions)
            infer_recs.append(_timing_rec(dict(policy.last_timing or {})))

            # Boundary-smoothing + IK-continuity state, persisted across chunks:
            #   start_idx — where to begin in the freshly received chunk (time-align)
            #   last_cmd  — last 16-vec actually commanded (JOINT space), for the ramp-in
            #   ik_q      — warm start for the next IK solve (continuity of the elbow DOF)
            start_idx = 0
            last_cmd = None
            ik_q = np.asarray(arm.get_arm_q(), dtype=np.float64)
            reset_requested = False
            for c in range(1, args.max_chunks + 1):
                H = actions.shape[0]
                end_idx = H if args.exec_steps <= 0 else min(start_idx + args.exec_steps, H)
                n = end_idx - start_idx
                if n <= 0:
                    raise RuntimeError(
                        f"chunk {c}: nothing left to execute (start_idx={start_idx}, "
                        f"horizon={H}) — --prefetch-lead too large for this horizon")
                lead = min(args.prefetch_lead, n)
                box: dict = {}
                th = None
                pending_skip = 0
                ik_max_m = 0.0

                exec_t0 = time.time()
                for i in range(n):
                    if arm.faulted():
                        raise RuntimeError("ArmController control thread faulted — aborting")
                    if keys.poll() == "r":
                        reset_requested = True
                        break
                    tic = time.time()

                    a_eef = actions[start_idx + i]
                    # EEF -> joints. Warm start from the previous solution so the
                    # redundant elbow DOF stays on one branch across the whole run.
                    ik_q, pos_err = kin.solve_ik(a_eef[LEFT_EEF_CHANNELS], a_eef[RIGHT_EEF_CHANNELS], ik_q)
                    ik_max_m = max(ik_max_m, pos_err)
                    if pos_err > IK_WARN_M:
                        log.warning(f"[chunk {c} step {i}] IK residual {pos_err*1e3:.1f} mm — "
                                    f"model predicted a barely-reachable EEF pose")
                    a = np.concatenate([ik_q, a_eef[[LEFT_GRIPPER_CHANNEL, RIGHT_GRIPPER_CHANNEL]]])
                    # Cross-fade the first --blend-steps from the last commanded pose.
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

                    # Fire the next-chunk request once `lead` steps remain so it overlaps.
                    if th is None and (n - i) <= lead:
                        obs_next = build_obs(cam, arm, grip, kin, prompt)
                        pending_skip = (n - 1 - i) if args.chunk_align else 0
                        th = threading.Thread(target=_infer_worker,
                                              args=(policy, obs_next, box),
                                              daemon=True, name=f"prefetch-{c}")
                        th.start()

                    sleep = dt - (time.time() - tic)
                    if sleep > 0:
                        time.sleep(sleep)

                if reset_requested:
                    # A prefetch may be in flight — join it before we reuse the
                    # policy socket, or two threads would talk on it at once.
                    if th is not None:
                        th.join()
                    break

                # Collect the prefetched next chunk (it should be done or nearly).
                exec_s = time.time() - exec_t0
                join_t0 = time.time()
                if th is not None:
                    th.join()
                join_wait_s = time.time() - join_t0
                if "err" in box:
                    raise box["err"]
                next_actions = apply_raisez(box["actions"], args.raisez)

                rec = _timing_rec(box.get("timing", {}))
                infer_recs.append(rec)
                chunk_recs.append({"exec_s": exec_s, "join_wait_s": join_wait_s, "ik_max_m": ik_max_m})
                log.info(f"[chunk {c}] execute={exec_s:.2f}s join_wait={join_wait_s*1e3:.0f}ms "
                         f"ik_max={ik_max_m*1e3:.2f}mm | "
                         f"infer wall={rec['wall_ms']:.0f}ms (gpu={rec['wait_recv_ms']:.0f}ms)")
                log_chunk_ranges(c, next_actions)
                actions = next_actions
                start_idx = min(pending_skip, next_actions.shape[0] - 1)

            if not reset_requested:
                break  # ran to --max-chunks — done

            # 'r' pressed: drop feedforward, ramp back to the ready pose, and wait
            # for Enter before starting a fresh inference session from the top.
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


# ---------- pipeline stages (kept identical to fastwam/main.py) ----------

def _release_grippers(grip, args) -> None:
    """Open both grippers so a reset lets go of whatever is being held.

    Runs BEFORE the arm ramps home, because _initialize_pose() below would
    otherwise (a) drag a still-gripped object across the table on its way to the
    ready pose and (b) clamp down harder — its gripper sequence closes to
    GRIPPER_MIN first. Blocks for --reset-gripper-duration while the arm's publish
    thread keeps holding the last commanded pose. Disable with
    --no-reset-open-gripper.
    """
    if not args.reset_open_gripper:
        return
    left, right = grip.get_state()
    log.info(f"Releasing: opening grippers L={left:.2f} R={right:.2f} -> "
             f"{args.reset_gripper_open:.2f} over {args.reset_gripper_duration:.1f}s")
    grip.move_to_targets(args.reset_gripper_open, args.reset_gripper_open,
                         duration=args.reset_gripper_duration)


def _initialize_pose(arm, grip, args) -> None:
    log.info(f"Moving arms to ready pose over {args.init_duration:.1f}s "
             f"(velocity_limit={args.velocity_limit} rad/s)")
    arm.move_to_pose(INIT_POSE_READY, duration=args.init_duration,
                     velocity_limit=args.velocity_limit)
    half = args.gripper_init_duration / 2
    log.info(f"Closing grippers to {GRIPPER_MIN} over {half:.1f}s")
    grip.move_to_targets(GRIPPER_MIN, GRIPPER_MIN, duration=half)
    log.info(f"Opening grippers to ({args.init_gripper_left}, {args.init_gripper_right}) "
             f"over {half:.1f}s")
    grip.move_to_targets(args.init_gripper_left, args.init_gripper_right, duration=half)
    log.info("Init complete.")
    time.sleep(args.settle_duration)
    log.info("Arms settled at ready pose.")


def _wait_for_operator(args) -> None:
    if args.auto_start:
        return
    log.info("===============================================================")
    log.info("STANDBY: arms locked at ready pose.")
    log.info("Set up the scene, then press [Enter] to start inference.")
    log.info("Press [Ctrl+C] at any time to abort safely.")
    log.info("===============================================================")
    try:
        input("")
    except EOFError:
        log.info("EOF on stdin — proceeding without prompt")


def _cleanup(arm, grip, cam, policy) -> None:
    """Release every resource. disable_arm_sdk MUST run — it returns arm
    authority to the locomotion service. Each step isolated so a second Ctrl+C
    cannot skip later steps."""
    log.info("Shutting down — releasing arm_sdk")
    try:
        arm.stop()
    except BaseException as e:
        log.warning(f"arm.stop() failed: {e}")
    try:
        arm.disable_arm_sdk()
    except BaseException as e:
        log.warning(f"disable_arm_sdk failed: {e}")
    if grip is not None:
        try:
            grip.stop()
        except BaseException as e:
            log.warning(f"grip.stop() failed: {e}")
    if cam is not None:
        try:
            cam.close()
        except BaseException as e:
            log.warning(f"cam.close() failed: {e}")
    if policy is not None:
        try:
            policy.close()
        except BaseException as e:
            log.warning(f"policy.close() failed: {e}")


def run(args) -> None:
    # EEF mode needs the kinematics for BOTH build_obs (measured q -> EEF state)
    # and the per-step IK, so it is always built (a bad URDF fails before DDS).
    # One instance is enough: FK, IK and gravity_torque all run on the main
    # thread; the prefetch worker only does policy.infer.
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
        # holding the arms at INIT_POSE_READY meanwhile — same as fastwam/main.py).
        policy = FastWAMPolicy(host=args.server_host, port=args.server_port)
        _run_inference_loop(arm, grip, cam, policy, kin, args)
        _initialize_pose(arm, grip, args)
    finally:
        _cleanup(arm, grip, cam, policy)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iface", required=True, help="Network interface to robot, e.g. enp0s31f6")
    p.add_argument("--server-host", required=True, help="FastWAM/serve_fastwam_g1.py host or IP")
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
                   help="How many action chunks to run before stopping")
    p.add_argument("--control-hz", type=float, default=30.0,
                   help="Per-step dispatch rate; match your LeRobot recording fps (30)")
    p.add_argument("--exec-steps", type=int, default=0,
                   help="Steps to execute per chunk before re-querying; 0 = full horizon.")
    p.add_argument("--prefetch-lead", type=int, default=5,
                   help="Start the next inference when this many steps remain in the "
                        "current chunk, so it overlaps and the boundary doesn't stall")
    p.add_argument("--blend-steps", type=int, default=5,
                   help="Cross-fade the first N steps of each new chunk from the last "
                        "commanded pose so a chunk swap ramps in smoothly (0 disables)")
    p.add_argument("--no-chunk-align", action="store_false", dest="chunk_align",
                   help="Disable chunk time-alignment (execute every chunk from index 0).")
    # ---- Gravity feedforward (reduced-arm model from openpi/eef_kinematics.py) ----
    p.add_argument("--tauff-scale", type=float, default=1.0,
                   help="Scale on the gravity-compensation feedforward torque fed to the "
                        "arm each step (default 1.0 = full comp, matching how the data was "
                        "collected). Use <1 (e.g. 0.5) for a cautious first pass, or 0 to "
                        "disable (the arm then sags and state feedback drifts OOD).")
    # ---- Safety / motion limits ----
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
