"""Masked / weighted-blend inference loop for the G1 against a FastWAM EEF policy
server (FastWAM/serve.py) — joint-space-merge variant of fastwam/main_eef.py.

WHAT CHANGED vs fastwam/main_eef.py
-----------------------------------
Same FastWAM transport (FastWAMPolicy), same obs wire format ({"image":[3 RGB],
"state":(16,), "prompt"}), same EEF action contract, gravity feedforward and
safety sequence — all reused verbatim (build_obs, apply_raisez, log_chunk_ranges,
the EEF channel layout, _KeyPoller / _initialize_pose / _wait_for_operator /
_cleanup). The ONLY new thing is the chunk-scheduling + blending policy in
`_run_masked_loop`: instead of executing a whole chunk then swapping (main_eef.py,
which only cross-fades the first --blend-steps of the new chunk), it commands
`lead + wait` steps per cycle and weighted-merges the freshly received chunk into
the plan over the WHOLE overlap. This is the FastWAM port of openpi/main_mask.py.

WHY THE MERGE IS IN JOINT SPACE
-------------------------------
Each EEF chunk is IK-solved to 14 joint targets *as soon as it arrives* (batch, on
the prefetch daemon thread so the IK overlaps the current window), and the plan is
stored in JOINT space. The weighted overlap merge (_merge_chunks) then blends
joints linearly — NOT EEF poses. Blending the EEF quaternions linearly across a
chunk swap would cut corners through SO(3); joint-space blending has no such
problem. This mirrors main_eef.py's per-step "blend after IK" rationale.

THREAD SAFETY
-------------
pinocchio's Data is not thread-safe, so the worker's batch IK must not share a
kinematics object with the main thread's FK (build_obs) + gravity_torque. Unlike
fastwam/main_eef.py (whose worker only calls policy.infer, so ONE instance is
enough), run() here builds TWO G1DualArmKinematics: `kin` for the main thread,
`kin_ik` for the worker. Each is only ever touched by one thread.

THE SCHEDULE
------------
A single merge cycle is `lead + wait` control steps (6 + 8 = 14 by default):

    command 6 steps from the current plan
      -> snapshot an obs and fire the next inference on a daemon thread
    command 8 more steps  (inference + batch IK run hidden behind this window)
      -> join: a fresh 32-step chunk arrives, already IK'd to joint space
      -> MERGE it into the plan, repeat

TUNING THE SCHEDULE (H = action horizon, 32 for a num_frames=33 FastWAM config)

    inference budget = wait / control_hz          seconds hidden behind execution
    merge overlap    = H - lead - 2*wait          steps the cross-fade spans
    hard requirement = lead + 2*wait < H          or the plan runs dry

The defaults (lead 6, wait 8, 30 Hz) give a 267 ms budget and a 10-step overlap.
FastWAM is a diffusion video model, so its sampler is much slower than an openpi
forward pass — check the `infer ok: ... in NNNms` line the server logs. If it
exceeds the budget, prefer LOWERING --control-hz (e.g. 15 Hz doubles the budget
to 533 ms and keeps the 10-step overlap, at the cost of playing the trajectory at
half the recording speed) over raising --infer-wait (which buys budget by eating
the overlap, so the mask degenerates toward a hard chunk swap). Cutting the
server's --num-inference-steps is the other lever. The end-of-run summary prints a
stall verdict either way.

TIME ALIGNMENT
--------------
The prefetch obs is taken `wait` steps before we adopt the new chunk, so the
chunk's first `wait` steps are already "in the past" by the time it lands. We drop
exactly those leading steps (`new_chunk[wait:]`) so the new prediction is
wall-clock aligned with the plan it is replacing — otherwise the arm snaps back to
a stale pose then forward again.

THE WEIGHTED MERGE  (the "mask")
--------------------------------
`old_future[k]` and `new_aligned[k]` are two joint predictions for the SAME future
timestep. We cross-fade them with a weight that ramps the new chunk in over the
overlap:

    a_new(k) = blend_start + (1 - blend_start) * k/(L-1)
    plan[k]  = (1 - a_new)*old_future[k] + a_new*new_aligned[k]

so the freshly predicted chunk eases in instead of snapping. Whatever of the new
chunk extends past the old plan (the non-overlapping tail) is appended verbatim.

Press 'r' at any time to ramp back to the ready pose and re-arm (wait for Enter
before a fresh session).

Precondition: robot already in 'ai' motion mode (set via the Unitree app), and
FastWAM/serve.py is serving an EEF-trained checkpoint.

Usage (run from the repo root):
  python fastwam/main_mask.py \\
      --iface enp0s31f6 \\
      --server-host 1.2.3.4 \\
      --server-port 8000 \\
      --prompt "pick the red bottle"
"""

import argparse
import logging
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import g1_client
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "openpi"))  # -> eef_kinematics

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from g1_client.arm_controller import ArmController
from g1_client.gripper_controller import GripperController, GRIPPER_MIN, GRIPPER_MAX
from g1_client.camera_client import CameraClient
from fastwam_policy import FastWAMPolicy

from eef_kinematics import G1DualArmKinematics, DEFAULT_URDF, DEFAULT_ASSETS

# Reuse everything that is identical to the EEF FastWAM client — controllers,
# EEF<->obs assembly, EEF channel layout, the keyboard poller and the
# init/cleanup/standby sequence are untouched; only the scheduling/blending
# policy below is new.
from main_eef import (
    LEFT_EEF_CHANNELS, RIGHT_EEF_CHANNELS,
    LEFT_GRIPPER_CHANNEL, RIGHT_GRIPPER_CHANNEL, IK_WARN_M,
    build_obs, apply_raisez, log_chunk_ranges,
    _pct, _stat, _timing_rec, _KeyPoller,
    _initialize_pose, _wait_for_operator, _cleanup,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("g1_fastwam.mask")

# The plan is stored in JOINT space after IK: [H, 16] = 14 joints + 2 grippers.
ARM_JOINTS = slice(0, 14)


# ---------- EEF chunk -> joint plan ----------

def _ik_chunk(eef_actions: np.ndarray, kin: G1DualArmKinematics,
              q_seed: np.ndarray):
    """Batch-IK a [H, 16] EEF chunk into a [H, 16] joint plan (14 joints + 2 grip).

    Each row is solved warm-started from the previous solution so the redundant
    elbow DOF stays on one branch across the whole chunk. Returns
    (joint_chunk, ik_max_m, last_q).
    """
    H = eef_actions.shape[0]
    out = np.empty((H, 16), dtype=np.float64)
    q = np.asarray(q_seed, dtype=np.float64)
    ik_max = 0.0
    for k in range(H):
        a = eef_actions[k]
        q, pos_err = kin.solve_ik(a[LEFT_EEF_CHANNELS], a[RIGHT_EEF_CHANNELS], q)
        ik_max = max(ik_max, pos_err)
        out[k, ARM_JOINTS] = q
        out[k, LEFT_GRIPPER_CHANNEL] = a[LEFT_GRIPPER_CHANNEL]
        out[k, RIGHT_GRIPPER_CHANNEL] = a[RIGHT_GRIPPER_CHANNEL]
    return out, ik_max, q


def _infer_ik_worker(policy: FastWAMPolicy, obs: dict, kin_ik: G1DualArmKinematics,
                     raisez: float, q_seed: np.ndarray, box: dict) -> None:
    """Infer one EEF chunk and batch-IK it to a joint plan on this daemon thread so
    both the network+GPU wait and the IK overlap the current execution window.

    Uses its OWN kinematics instance (kin_ik) so it never touches the main thread's
    kin (build_obs FK + gravity_torque) concurrently — pinocchio Data is not
    thread-safe. Result/exception surfaced via `box`.
    """
    try:
        result = policy.infer(obs)
        box["timing"] = dict(policy.last_timing or {})   # pack/send/wait_recv/unpack (ms via _timing_rec)
        ik_t0 = time.time()
        eef = np.asarray(result["actions"], dtype=np.float64)
        if eef.ndim != 2 or eef.shape[1] < 16:
            raise RuntimeError(f"Unexpected action shape {eef.shape} (want [H, 16])")
        eef = apply_raisez(eef, raisez)
        joint, ik_max, _ = _ik_chunk(eef, kin_ik, q_seed)
        box["ik_ms"] = (time.time() - ik_t0) * 1e3        # client-side batch IK cost
        box["eef"] = eef              # kept for EEF-range logging on the main thread
        box["actions"] = joint        # the joint plan the merge consumes
        box["ik_max_m"] = ik_max
    except BaseException as e:  # surface to the main thread
        box["err"] = e


# ---------- the weighted overlap merge (the "mask") ----------

def _merge_chunks(old_future: np.ndarray, new_aligned: np.ndarray,
                  blend_start: float = 0.0) -> np.ndarray:
    """Weighted temporal blend of the old plan tail and the time-aligned new chunk.

    Both are JOINT-space [., 16] predictions for the same future timesteps. The
    new-chunk weight ramps from `blend_start` at the first overlap step to 1.0 at
    the last, so the new chunk fades in. `blend_start=0.0` makes the join seamless
    (the first merged step equals the old plan exactly — no boundary jump), which
    is the main knob against boundary 回抽; `blend_start=0.5` is a 1:1 crossfade.
    The longer chunk's non-overlapping tail is appended unblended.

    In this loop's cadence new_aligned is always >= old_future, so the appended
    tail is the new chunk's extra steps — but the old-extends-further branch is
    kept so a slow inference (short new chunk) can't drop planned steps.
    """
    Lo, Ln = len(old_future), len(new_aligned)
    L = min(Lo, Ln)
    out = np.empty((max(Lo, Ln), old_future.shape[1]), dtype=np.float64)
    for k in range(L):
        a_new = 1.0 if L == 1 else blend_start + (1.0 - blend_start) * (k / (L - 1))
        out[k] = (1.0 - a_new) * old_future[k] + a_new * new_aligned[k]
    if Ln > L:
        out[L:] = new_aligned[L:]
    elif Lo > L:
        out[L:] = old_future[L:]
    return out


# ---------- latency profiling ----------

def _summarize_timing(infer_recs, merge_recs, args) -> None:
    """End-of-run latency breakdown: per-infer components (pack/send/wait_recv/
    unpack), the bottleneck, and the per-merge stall verdict — the same picture
    fastwam/main_eef.py prints, plus the client-side batch-IK cost this mask loop
    adds.

    wait_recv is the blocking recv = network round-trip + the server's diffusion
    sampler combined. For FastWAM the sampler dominates, so this is mostly GPU
    time; serve.py does not report its own infer time in the payload (it only logs
    it), so the two cannot be split here — compare against the server's
    `infer ok: ... in NNNms` log line.
    """
    if not infer_recs:
        return
    budget_ms = (args.infer_wait / args.control_hz) * 1e3
    log.info("=" * 64)
    comps = [("pack", "pack_ms"), ("send", "send_ms"),
             ("wait_recv(gpu)", "wait_recv_ms"), ("unpack", "unpack_ms"),
             ("wall(total)", "wall_ms")]
    log.info(f"per-infer latency over {len(infer_recs)} calls (ms):")
    log.info(f"  {'component':<16} {'min':>7} {'p50':>7} {'p95':>7} {'max':>7} {'mean':>7}")
    means = {}
    for label, key in comps:
        mn, p50, p95, mx, me = _stat([r[key] for r in infer_recs])
        means[label] = me
        log.info(f"  {label:<16} {mn:7.1f} {p50:7.1f} {p95:7.1f} {mx:7.1f} {me:7.1f}")
    sub = {k: means[k] for k in ("pack", "send", "wait_recv(gpu)", "unpack")}
    bn = max(sub, key=sub.get)
    log.info(f"BOTTLENECK (mean): {bn} = {sub[bn]:.1f} ms "
             f"({sub[bn]/means['wall(total)']*100:.0f}% of infer total)")
    up = [r["bytes_sent"] for r in infer_recs if r["bytes_sent"] > 0]
    if up:
        log.info(f"  upload payload ~= {np.mean(up)/1024:.0f} KiB/infer "
                 f"(3 decoded RGB views — the FastWAM obs format; the server resizes)")
    if merge_recs:
        jw = [r["join_wait_s"] * 1e3 for r in merge_recs]
        ik = [r["ik_ms"] for r in merge_recs]
        ikm = [r["ik_max_m"] * 1e3 for r in merge_recs]
        stalled = sum(1 for x in jw if x > 1.0)
        log.info(f"merge join_wait (ms): mean={np.mean(jw):.0f} p95={_pct(jw,95):.0f} "
                 f"max={max(jw):.0f} | stalled {stalled}/{len(jw)} "
                 f"(overlap budget {budget_ms:.0f}ms)")
        log.info(f"client IK build/chunk (ms): mean={np.mean(ik):.0f} p95={_pct(ik,95):.0f} "
                 f"max={max(ik):.0f}")
        log.info(f"IK residual/chunk (mm): mean={np.mean(ikm):.2f} p95={_pct(ikm,95):.2f} "
                 f"max={max(ikm):.2f}")
        if stalled:
            log.warning(f"{stalled} merge(s) STALLED — infer+IK didn't fit the {budget_ms:.0f}ms "
                        f"overlap window, so time-alignment breaks and the arm 回抽. Lower "
                        f"--control-hz (keeps the merge overlap), cut the server's "
                        f"--num-inference-steps, or raise --infer-wait (shrinks the overlap).")
        else:
            log.info("no merge stalled — infer+IK fully hidden behind execution.")
    log.info("=" * 64)


# ---------- inference loop ----------

def _run_masked_loop(arm, grip, cam, policy, kin, kin_ik, args) -> None:
    """Receding-horizon loop: command `lead` steps, prefetch (infer + batch IK),
    command `wait` steps, then merge the prefetched joint chunk with a weighted
    overlap blend.

    The arm/gripper publish threads keep holding the last commanded target while
    `th.join()` blocks, so even if FastWAM's sampler overruns the `wait` window the
    worst case is a brief hold plus a time-alignment error, not an unsafe state.

    Press 'r' at any time to ramp back to the ready pose and re-arm. Gravity
    feedforward holds each commanded pose against gravity when --tauff-scale > 0.
    """
    dt = 1.0 / args.control_hz
    prompt = args.prompt
    lead = args.exec_before_prefetch          # steps before firing the prefetch
    wait = args.infer_wait                    # steps the inference overlaps with
    cycle = lead + wait                       # steps commanded per merge
    skip = wait if args.time_align else 0     # stale leading steps to drop
    log.info(f"schedule: lead={lead} wait={wait} cycle={cycle} @ {args.control_hz:g}Hz "
             f"-> inference budget {wait/args.control_hz*1e3:.0f}ms/merge, "
             f"merge overlap H-{lead + 2*wait} steps")
    if args.tauff_scale > 0:
        log.info(f"gravity feedforward ON (--tauff-scale {args.tauff_scale}): the arm "
                 f"holds commanded poses against gravity, matching collection dynamics")
    else:
        log.warning("gravity feedforward OFF (--tauff-scale 0): the arm will sag below "
                    "commanded poses — state feedback drifts out of the training distribution")

    infer_recs = []   # one per inference (first + each prefetch): latency breakdown
    merge_recs = []   # one per merge: join_wait, client IK time, IK residual
    with _KeyPoller() as keys:
        log.info("Press [r] at any time to reset to the ready pose and re-arm.")
        while True:
            # First chunk is a blocking infer + IK (nothing to overlap yet).
            log.info(f"First inference (prompt={prompt!r}) — FastWAM warm-up may be slow")
            result = policy.infer(build_obs(cam, arm, grip, kin, prompt))
            eef = np.asarray(result["actions"], dtype=np.float64)
            if eef.ndim != 2 or eef.shape[1] < 16:
                raise RuntimeError(f"Unexpected action shape {eef.shape} (want [H, 16])")
            if args.raisez:
                log.info(f"--raisez {args.raisez:.1f} mm: offsetting every EEF Z target by "
                         f"{args.raisez*1e-3:+.4f} m (pelvis frame)")
            eef = apply_raisez(eef, args.raisez)
            log_chunk_ranges(0, eef)
            plan, ik_max, _ = _ik_chunk(eef, kin_ik, np.asarray(arm.get_arm_q(), dtype=np.float64))
            if ik_max > IK_WARN_M:
                log.warning(f"[chunk 0] IK residual {ik_max*1e3:.1f} mm — model predicted "
                            f"a barely-reachable EEF pose")
            if plan.shape[0] <= skip + cycle:
                raise RuntimeError(f"horizon {plan.shape[0]} too short for lead={lead} "
                                   f"wait={wait} (need lead + 2*wait < H, i.e. > "
                                   f"{skip + cycle}) — lower --infer-wait/"
                                   f"--exec-before-prefetch")
            infer_recs.append(_timing_rec(dict(policy.last_timing or {})))

            ptr = 0              # index of the next step to command within `plan`
            since_merge = 0      # steps commanded since the last merge
            prefetch_fired = False
            box: dict = {}
            th = None
            reset_requested = False

            for merges in range(1, args.max_chunks + 1):
                # Command `cycle` steps from the current plan, firing the prefetch
                # `wait` steps before the end of the window.
                for _ in range(cycle):
                    if arm.faulted():
                        raise RuntimeError("ArmController control thread faulted — aborting")
                    if keys.poll() == "r":
                        reset_requested = True
                        break
                    if ptr >= len(plan):
                        raise RuntimeError(f"plan exhausted (ptr={ptr}, len={len(plan)}) — "
                                           f"inference slower than {wait} steps")
                    tic = time.time()

                    a = plan[ptr]
                    arm.set_arm_target(a[ARM_JOINTS])
                    # Gravity feedforward: hold the commanded pose instead of sagging
                    # under kp — matches the collection-time dynamics (tau=sol_tauff) the
                    # policy was trained on, so state feedback stays in-distribution.
                    if args.tauff_scale > 0:
                        arm.set_arm_tauff(kin.gravity_torque(a[ARM_JOINTS], args.tauff_scale))
                    grip.set_targets(
                        float(np.clip(a[LEFT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
                        float(np.clip(a[RIGHT_GRIPPER_CHANNEL], GRIPPER_MIN, GRIPPER_MAX)),
                    )
                    ptr += 1
                    since_merge += 1

                    # After `lead` steps, snapshot an obs and fire the next
                    # inference+IK so it overlaps the remaining `wait` steps. The
                    # worker warm-starts IK from the joints we just commanded.
                    if since_merge == lead and not prefetch_fired:
                        obs_next = build_obs(cam, arm, grip, kin, prompt)
                        box = {}
                        th = threading.Thread(
                            target=_infer_ik_worker,
                            args=(policy, obs_next, kin_ik, args.raisez, a[ARM_JOINTS].copy(), box),
                            daemon=True, name=f"prefetch-{merges}")
                        th.start()
                        prefetch_fired = True

                    sleep = dt - (time.time() - tic)
                    if sleep > 0:
                        time.sleep(sleep)

                if reset_requested:
                    # A prefetch may be in flight — join it before we reuse the
                    # policy socket, or two threads would talk on it at once.
                    if th is not None:
                        th.join()
                    break

                # Adopt the prefetched joint chunk and merge it into the plan.
                join_t0 = time.time()
                th.join()
                join_wait_ms = (time.time() - join_t0) * 1e3
                if "err" in box:
                    raise box["err"]
                new_chunk = box["actions"]            # joint [H, 16], row 0 ~ obs@(merge-wait)
                new_aligned = new_chunk[skip:]        # drop the `wait` already-elapsed steps
                old_future = plan[ptr:]               # what the old plan still has queued
                plan = _merge_chunks(old_future, new_aligned, args.blend_start)
                ik_max = box["ik_max_m"]
                ptr = 0
                since_merge = 0
                prefetch_fired = False
                th = None

                rec = _timing_rec(box.get("timing", {}))
                ik_ms = box.get("ik_ms", 0.0)
                infer_recs.append(rec)
                merge_recs.append({"join_wait_s": join_wait_ms / 1e3,
                                   "ik_ms": ik_ms, "ik_max_m": ik_max})

                overlap = min(len(old_future), len(new_aligned))
                appended = max(0, len(new_aligned) - len(old_future))
                log.info(f"[merge {merges}] overlap={overlap} append={appended} "
                         f"plan_len={len(plan)} ik_max={ik_max*1e3:.2f}mm "
                         f"join_wait={join_wait_ms:.0f}ms"
                         + (" STALLED" if join_wait_ms > 1.0 else "")
                         + f" | infer wall={rec['wall_ms']:.0f}ms "
                         + f"gpu={rec['wait_recv_ms']:.0f}ms ik_build={ik_ms:.0f}ms"
                         + (f" up={rec['bytes_sent']/1024:.0f}KiB" if rec['bytes_sent'] > 0 else ""))
                if ik_max > IK_WARN_M:
                    log.warning(f"[merge {merges}] IK residual {ik_max*1e3:.1f} mm — model "
                                f"predicted a barely-reachable EEF pose")
                log_chunk_ranges(merges, box["eef"])

            if not reset_requested:
                break  # ran to --max-chunks — done

            # 'r' pressed: drop feedforward, ramp back to the ready pose, and wait
            # for Enter before starting a fresh inference session from the top.
            log.info("[r] reset requested — returning to ready pose")
            if args.tauff_scale > 0:
                arm.set_arm_tauff(np.zeros(14))
            _initialize_pose(arm, grip, args)
            log.info("Press [Enter] to resume inference, or [Ctrl+C] to abort.")
            keys.wait_enter()

    # Drop the feedforward before run() ramps back to the ready pose, so that
    # move runs with the arm's default (tau=0) dynamics.
    if args.tauff_scale > 0:
        arm.set_arm_tauff(np.zeros(14))
    _summarize_timing(infer_recs, merge_recs, args)


# ---------- entry point ----------

def run(args) -> None:
    # Two kinematics instances: `kin` for the main thread (build_obs FK + gravity
    # tauff), `kin_ik` for the prefetch worker's batch IK. pinocchio Data is not
    # thread-safe, so they must not be shared. A bad URDF fails here, before DDS.
    log.info(f"Loading G1 dual-arm model from {args.urdf}")
    kin = G1DualArmKinematics(args.urdf, args.assets)
    kin_ik = G1DualArmKinematics(args.urdf, args.assets)

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
        _run_masked_loop(arm, grip, cam, policy, kin, kin_ik, args)
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
                   help="How many merge cycles to run before stopping")
    p.add_argument("--control-hz", type=float, default=30.0,
                   help="Per-step dispatch rate; match your LeRobot recording fps (30). "
                        "Lowering it is the best lever when FastWAM's sampler overruns "
                        "the prefetch window — it buys budget without eating the overlap.")
    # ---- Masked-blend schedule (horizon-32 / num_frames-33 defaults) ----
    p.add_argument("--exec-before-prefetch", type=int, default=6,
                   help="Steps to command before firing the next inference (default 6)")
    p.add_argument("--infer-wait", type=int, default=8,
                   help="Steps to command while the inference runs; also the number of "
                        "stale leading steps dropped for time-alignment (default 8). "
                        "Must satisfy exec_before_prefetch + 2*infer_wait < horizon; "
                        "raising it buys inference budget but shrinks the merge overlap.")
    p.add_argument("--no-time-align", action="store_false", dest="time_align",
                   help="Do not drop the new chunk's leading `infer_wait` steps. By "
                        "default they are dropped so the new chunk is wall-clock "
                        "aligned with the plan it replaces; pass this for A/B testing.")
    p.add_argument("--blend-start", type=float, default=0.0,
                   help="New-chunk weight at the FIRST overlap step of the mask, ramping "
                        "to 1.0 at the last. 0.0 (default) = seamless join (start fully "
                        "on the old plan, fade to new) — least boundary 回抽; 0.5 = a 1:1 "
                        "crossfade. Lower is smoother at the cost of the new prediction "
                        "taking effect a few steps later.")
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
    p.add_argument("--auto-start", action="store_true",
                   help="Skip the post-init Enter prompt and start immediately.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
