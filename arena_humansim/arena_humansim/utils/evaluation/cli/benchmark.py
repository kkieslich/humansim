import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from arena_humansim.params import WAYPOINT_MODES
from arena_humansim.utils.evaluation.cli._resume import latest_sweep_dir, peds_root
from arena_humansim.utils.evaluation.cli.sweep import _trial_dir_name
from arena_humansim.utils.scenario_discovery import scenario_roots


def _discover_robot_policies() -> list[str]:
    from arena_humansim.local_planner import LocalPlanner

    return sorted(name for name, info in LocalPlanner.info().items() if info.robot_policy)


def _discover_ped_planners() -> list[str]:
    from arena_humansim.local_planner import LocalPlanner

    return sorted(name for name, info in LocalPlanner.info().items() if not info.robot_policy)


def _discover_scenarios(robots_mode: bool) -> list[str]:
    # Returns relative paths (e.g. `nav/sparse/corridor`) so the launch resolver
    # doesn't have to disambiguate same-stem files across density tiers.
    # robots_mode=True  -> every robot variant (`robot_*.yaml`) plus all of `het/` (constitutive).
    # robots_mode=False -> every pure-ped scenario (no `robot_` prefix, excluding `het/`).
    out: set[str] = set()
    for root in scenario_roots():
        if root.name != "evaluation":
            continue
        for yaml_path in root.glob("**/*.yaml"):
            rel = yaml_path.relative_to(root).with_suffix("")
            in_het = "het" in rel.parts
            is_robot_variant = yaml_path.stem.startswith("robot_")
            if robots_mode:
                if is_robot_variant or in_het:
                    out.add(str(rel))
            else:
                if not is_robot_variant and not in_het:
                    out.add(str(rel))
    return sorted(out)


MAX_WORKERS = 50


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel sweep + monitor + analyze. Add --robot_policies for a robot-policy sweep; otherwise runs the pedestrian divergence sweep.")
    parser.add_argument("--scenarios", nargs="+", default=None, help="Scenario relative paths under config/evaluation/ (e.g. nav/dense/corridor). Default = auto-discovered: pure-ped scenarios in divergence mode, robot variants + all het/ scenarios when --robot_policies is set.")
    parser.add_argument("--planners", nargs="*", default=None, help="Pedestrian local planners. Omit or pass --planners with no values -> auto-discover all (registry minus local_planner/robot/). Pass --planners a b c -> use those.")
    parser.add_argument(
        "--robot_policies",
        nargs="*",
        default=None,
        help="Robot local planners. Omit the flag entirely -> divergence sweep (no robot trials). Pass --robot_policies with no values -> robots mode using every policy under local_planner/robot/ (auto-discovered). Pass --robot_policies a b c -> robots mode with the listed policies.",
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 1) // 2, 1))
    parser.add_argument("--sim_duration", type=int, default=60)
    parser.add_argument("--force_planner_scenarios", nargs="*", default=[])
    parser.add_argument("--run_dir", type=str, default=None, help="Output dir; defaults to $ARENA_DATA_DIR/peds/<ts>_sweep. The suffix is convention only - `--resume` finds any dir under peds/ containing a run_args.json manifest.")
    parser.add_argument("--resume", action="store_true", help="Resume the latest incomplete run under $ARENA_DATA_DIR/peds; mode (divergence/robots) is reconstructed from the stored manifest.")
    parser.add_argument("--cop", action="store_true", help="Mark this as a cognitive-object-permanence ablation run. Requires --robot_policies. Recorded in the manifest as cop=true so the COP analyzer can gate on it.")
    parser.add_argument(
        "--force_waypoint_mode",
        choices=WAYPOINT_MODES.values,
        default="reverse",
        help="Override scenario waypoint_mode for every kind=human agent (robots untouched). Default 'reverse' keeps single-waypoint pedestrians cycling for the full sim_duration instead of freezing at the goal. Pass '' to use scenario values.",
    )
    parser.add_argument("--trial_timeout_factor", type=float, default=3.0, help="kill a trial after factor x sim_duration seconds of wall clock (0 disables).")
    parser.add_argument("--prune", action="store_true", help="delete <trial>/bag after extraction unless the trial failed the integrity gate or falls in the deterministic 5%% keep sample.")
    args = parser.parse_args()

    if args.robot_policies is None:
        robot_policies: list[str] = []
        robots_mode = False
    else:
        robot_policies = [rp for rp in args.robot_policies if rp]
        if not robot_policies:
            robot_policies = _discover_robot_policies()
            if not robot_policies:
                parser.error("--robot_policies passed without values, but no robot policies discovered under local_planner/robot/")
            print(f"Auto-discovered robot policies: {robot_policies}")
        robots_mode = True
    if args.cop and not robots_mode:
        parser.error("--cop requires --robot_policies (COP is a property of the robot policy under test)")
    mode = "robots" if robots_mode else "divergence"
    args.robot_policies = robot_policies

    explicit_planners = [p for p in args.planners if p] if args.planners else []
    if not explicit_planners:
        args.planners = _discover_ped_planners()
        if not args.planners:
            parser.error("no pedestrian planners discovered in the local_planner registry")
        print(f"Auto-discovered pedestrian planners: {args.planners}")
    else:
        args.planners = explicit_planners

    if args.scenarios is None:
        args.scenarios = _discover_scenarios(robots_mode)
        if not args.scenarios:
            parser.error("no scenarios discovered under config/evaluation/. Pass --scenarios explicitly or check the install.")

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    elif args.resume:
        root = peds_root()
        if root is None:
            parser.error("--resume requires $ARENA_DATA_DIR (or pass --run_dir explicitly)")
        found = latest_sweep_dir(incomplete_only=True)
        if found is None:
            sys.exit(f"--resume: no incomplete run found under {root}")
        run_dir, _stored, done, expected = found
        print(f"Resuming {run_dir} ({done}/{expected} trials done).")
    else:
        root = peds_root()
        if root is None:
            parser.error("--run_dir required when $ARENA_DATA_DIR is unset")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = root / f"{ts}_sweep"

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_args.json"
    # Trial-defining args must stay consistent across resumes; runtime knobs
    # (workers, resume) come from the current invocation.
    # `mode` and `cop` are run-defining markers that downstream analyzers gate on.
    trial_keys = {"scenarios", "planners", "robot_policies", "seeds", "sim_duration", "force_planner_scenarios", "mode", "cop", "force_waypoint_mode"}
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text())
        print(f"Resuming run from {run_dir}; using stored run_args.json for trial set.")
        for k, v in stored.items():
            if k in trial_keys:
                setattr(args, k, v)
        robot_policies = [rp for rp in (args.robot_policies or []) if rp]
        robots_mode = stored.get("mode") == "robots" if "mode" in stored else bool(robot_policies)
        args.cop = bool(stored.get("cop", False))
        # Pre-existing runs predate force_waypoint_mode; keep their behavior identical.
        args.force_waypoint_mode = str(stored.get("force_waypoint_mode", ""))
    else:
        args.mode = mode
        stored = {k: v for k, v in vars(args).items() if k in trial_keys}
        manifest_path.write_text(json.dumps(stored, indent=2, sort_keys=True))

    log_dir = run_dir / "logs"
    progress_dir = run_dir / "progress"
    trials_dir = run_dir / "trials"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)

    print(f"Recordings: {run_dir}")
    print(f"Logs: {log_dir}")
    if robots_mode:
        cop_tag = " (COP ablation)" if args.cop else ""
        print(f"Robots mode{cop_tag}: robot_policies={robot_policies}")

    if robots_mode:
        all_trials = [(s, p, rp, seed) for s in args.scenarios for p in args.planners for rp in robot_policies for seed in range(1, args.seeds + 1)]
    else:
        all_trials = [(s, p, "", seed) for s in args.scenarios for p in args.planners for seed in range(1, args.seeds + 1)]
    remaining = [t for t in all_trials if not (run_dir / _trial_dir_name(*t) / ".done").exists()]
    if not remaining:
        print(f"All {len(all_trials)} trials already complete.")
    elif len(remaining) < len(all_trials):
        print(f"Resuming: {len(all_trials) - len(remaining)}/{len(all_trials)} trials already done; {len(remaining)} remaining.")

    claims_dir = run_dir / ".claims"
    if claims_dir.exists():
        for stale in claims_dir.glob("*.lock"):
            stale.unlink()

    if args.workers > MAX_WORKERS:
        print(f"Clamping workers from {args.workers} to {MAX_WORKERS} (ROS_DOMAIN_ID range).")
    workers = min(args.workers, len(remaining), MAX_WORKERS) if remaining else 0

    failed = 0
    n_workers_started = 0
    if remaining:
        trial_file = trials_dir / "all.txt"
        if robots_mode:
            trial_file.write_text("".join(f"{s}:{p}:{rp}:{seed}\n" for s, p, rp, seed in remaining))
        else:
            trial_file.write_text("".join(f"{s}:{p}:{seed}\n" for s, p, _rp, seed in remaining))

        worker_procs: list[subprocess.Popen[bytes]] = []
        progress_files: list[str] = []
        log_files = []
        for i in range(workers):
            progress_file = progress_dir / f"worker_{i}.txt"
            progress_files.append(str(progress_file))
            log_file = log_dir / f"worker_{i}.log"

            cmd = [
                sys.executable,
                "-m",
                "arena_humansim.utils.evaluation.cli",
                "sweep",
                "--trial_file",
                str(trial_file),
                "--sim_duration",
                str(args.sim_duration),
                "--output_dir",
                str(run_dir),
                "--worker_id",
                str(i),
                "--num_workers",
                str(workers),
                "--progress_file",
                str(progress_file),
            ]
            if args.force_planner_scenarios:
                cmd.extend(["--force_planner_scenarios", *args.force_planner_scenarios])
            if robots_mode:
                cmd.extend(["--robot_shutdown", "true"])
            if args.force_waypoint_mode:
                cmd.extend(["--force_waypoint_mode", args.force_waypoint_mode])
            cmd.extend(["--trial_timeout_factor", str(args.trial_timeout_factor)])
            if args.prune:
                cmd.append("--prune")

            logf = open(log_file, "wb")
            log_files.append(logf)
            proc = subprocess.Popen(cmd, stdout=logf, stderr=logf)
            worker_procs.append(proc)
            d_lo, d_hi = i * 2 + 2, i * 2 + 3
            own_shard_size = sum(1 for j, _ in enumerate(remaining) if j % workers == i)
            print(f"Worker {i} started (PID {proc.pid}): {own_shard_size} initial trials, ROS_DOMAIN_ID in {{{d_lo},{d_hi}}}")

        n_workers_started = len(worker_procs)
        print()
        monitor_cmd = [
            sys.executable,
            "-m",
            "arena_humansim.utils.evaluation.cli",
            "monitor",
            *progress_files,
            "--total_done",
            str(len(all_trials) - len(remaining)),
            "--total_grand",
            str(len(all_trials)),
        ]
        monitor_proc = subprocess.Popen(monitor_cmd)

        for proc in worker_procs:
            if proc.wait() != 0:
                failed += 1

        try:
            monitor_proc.send_signal(signal.SIGTERM)
            monitor_proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            monitor_proc.kill()

        for logf in log_files:
            logf.close()

    print()
    print("=== Sweep complete ===")
    expected = len(all_trials)
    done = sum(1 for _ in run_dir.rglob(".done"))
    print(f"Trials complete: {done}/{expected}")
    if failed > 0:
        print(f"WARNING: {failed}/{n_workers_started} workers exited nonzero (see {log_dir}/worker_*.log)")
    print(f"Recordings: {run_dir}")
    print()

    if done < expected:
        print(f"Skipping analysis: {expected - done} trials remaining. Re-invoke with --resume to continue.")
        return

    print("Run `python3 -m arena_humansim.utils.evaluation.cli analyze` to analyze this dir.")


if __name__ == "__main__":
    main()
