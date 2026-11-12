import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

FINAL_RTF_RE = re.compile(r"sim=([0-9.]+)s, wall=([0-9.]+)s, compute=([0-9.]+)s")

TIMEOUT_RC = 124
STUB_BAG_BYTES = 100_000
MIN_BAG_FRAMES = 100
MIN_FREE_BYTES = 20 * 1024**3
BAG_SAMPLE_RATE = 0.05

_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


from arena_humansim.params import WAYPOINT_MODES
from arena_humansim.utils.scenario_discovery import scenario_has_robot as _scenario_has_robot


def _write_progress(progress_file: Path | None, current: int, total: int) -> None:
    if progress_file is None:
        return
    tmp = progress_file.with_suffix(progress_file.suffix + ".tmp")
    tmp.write_text(f"{current}/{total}\n")
    tmp.replace(progress_file)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _kill_group(proc: subprocess.Popen, grace: float = 15.0) -> None:
    """SIGINT the launch process group so the recorder closes its bag, then escalate until the whole group is gone."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if proc.poll() is not None and not _group_alive(pgid):
                return
            time.sleep(0.1)


def _run_and_tee(cmd: list[str], env: dict[str, str], timeout: float | None = None) -> tuple[float, float | None, float | None, float | None, int]:
    """Tee stdout while scraping the final RTF line, so the deadline wraps proc.wait rather than subprocess.run."""
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    scraped: dict[str, float] = {}
    assert proc.stdout is not None

    def _pump() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            sys.stdout.write(line)
            m = FINAL_RTF_RE.search(line)
            if m:
                scraped["sim"] = float(m.group(1))
                scraped["wall"] = float(m.group(2))
                scraped["compute"] = float(m.group(3))

    pump = threading.Thread(target=_pump, daemon=True)
    pump.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        print(f"Trial exceeded the {timeout:.0f}s wall-clock budget, killing the launch process group.")
        _kill_group(proc)
    pump.join(timeout=10)
    if not pump.is_alive():
        proc.stdout.close()

    total_wall = time.monotonic() - t0
    rc = TIMEOUT_RC if timed_out else proc.returncode
    return total_wall, scraped.get("sim"), scraped.get("wall"), scraped.get("compute"), rc


def _try_claim(claims_dir: Path, trial_name: str) -> bool:
    # O_CREAT|O_EXCL atomically; the first worker to create the file wins.
    try:
        (claims_dir / f"{trial_name}.lock").touch(exist_ok=False)
        return True
    except FileExistsError:
        return False


def _keep_bag_sample(trial_id: str, rate: float = BAG_SAMPLE_RATE) -> bool:
    """Deterministic sample keyed on the trial id."""
    h = int(hashlib.sha256(trial_id.encode()).hexdigest()[:12], 16)
    return (h % 1_000_000) < int(rate * 1_000_000)


def _bag_bytes(trial_dir: Path) -> int:
    bag = trial_dir / "bag"
    if not bag.is_dir():
        return 0
    return sum(p.stat().st_size for p in bag.rglob("*") if p.is_file())


def _run_extractor(trial_dir: Path) -> tuple[bool, str]:
    """Run the extractor named by ARENA_SWEEP_EXTRACTOR, which writes <trial>/_extracted.json."""
    script = os.environ.get("ARENA_SWEEP_EXTRACTOR", "")
    if not script or not Path(script).exists():
        return False, "no extractor configured (set ARENA_SWEEP_EXTRACTOR)"
    python = os.environ.get("ARENA_SWEEP_EXTRACTOR_PYTHON") or sys.executable
    out = trial_dir / "_extracted.json"
    cmd = [python, script, str(trial_dir), "--out", str(out)]
    env = {**os.environ, **_THREAD_ENV, "PYTHONPATH": str(Path(script).parent) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return False, "extractor timed out"
    if proc.returncode != 0:
        return False, f"extractor rc={proc.returncode}: {proc.stderr.strip()[-400:]}"
    if not out.exists() or out.stat().st_size == 0:
        return False, "extractor produced no output"
    try:
        first = json.loads(out.read_text().splitlines()[0])
    except (ValueError, IndexError) as exc:
        return False, f"extractor output is not JSON: {exc}"
    if first.get("ok") is False:
        return False, f"extraction failed: {first.get('error', 'unknown')}"
    frames = first.get("n_frames")
    if isinstance(frames, int) and frames < MIN_BAG_FRAMES:
        return False, f"only {frames} recorded frames (< {MIN_BAG_FRAMES}), the node died early"
    return True, ""


def _integrity(trial_dir: Path, rc: int) -> tuple[bool, list[str]]:
    """Gate on the launch rc, the recording manifest and the bag size."""
    problems: list[str] = []
    if rc != 0:
        problems.append(f"launch rc={rc}")
    manifest = trial_dir / "recording.yaml"
    if manifest.exists():
        try:
            info = yaml.safe_load(manifest.read_text()) or {}
        except (OSError, yaml.YAMLError) as exc:
            info = {}
            problems.append(f"unreadable recording.yaml: {exc}")
        if info.get("contaminated"):
            problems.append(f"contaminated: publishers={info.get('publishers')}")
        if not info.get("finished_at"):
            problems.append("recorder never closed cleanly (no finished_at)")
        for topic, count in (info.get("publishers") or {}).items():
            if isinstance(count, int) and count > 1:
                problems.append(f"{topic} had {count} publishers")
    else:
        problems.append("no recording.yaml")
    nbytes = _bag_bytes(trial_dir)
    if nbytes < STUB_BAG_BYTES:
        problems.append(f"stub bag ({nbytes} bytes), the node probably died early")
    extra = os.environ.get("ARENA_SWEEP_INTEGRITY", "")
    if extra and Path(extra).exists():
        python = os.environ.get("ARENA_SWEEP_EXTRACTOR_PYTHON") or sys.executable
        proc = subprocess.run([python, extra, str(trial_dir)], capture_output=True, text=True)
        (trial_dir / "_integrity.json").write_text(proc.stdout)
        if proc.returncode != 0:
            problems.append(f"integrity script rc={proc.returncode}")
    return (not problems), problems


def _prune_bag(trial_dir: Path, trial_id: str, integrity_ok: bool) -> str:
    if not integrity_ok:
        return "kept: integrity failure"
    if _keep_bag_sample(trial_id):
        return "kept: sample"
    bag = trial_dir / "bag"
    if bag.is_dir():
        shutil.rmtree(bag, ignore_errors=True)
        return "pruned"
    return "no bag"


def _free_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return MIN_FREE_BYTES


def run_sweep(
    trials: list[tuple[str, str, str, int]],
    sim_duration: int,
    output_dir: Path,
    force_planner_scenarios: set[str] | frozenset[str] = frozenset(),
    worker_id: int = 0,
    num_workers: int = 1,
    progress_file: Path | None = None,
    robot_shutdown: str = "",
    force_waypoint_mode: str = "",
    trial_timeout_factor: float = 3.0,
    prune: bool = False,
) -> None:
    if robot_shutdown not in ("", "true", "false"):
        raise ValueError(f"robot_shutdown must be '', 'true', or 'false'; got {robot_shutdown!r}")
    if force_waypoint_mode not in WAYPOINT_MODES.values:
        raise ValueError(f"force_waypoint_mode must be one of {WAYPOINT_MODES.values}; got {force_waypoint_mode!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if progress_file is not None:
        progress_file.parent.mkdir(parents=True, exist_ok=True)

    claims_dir = output_dir / ".claims"
    claims_dir.mkdir(parents=True, exist_ok=True)

    if worker_id == 0:
        robot_scenarios = {scenario for scenario, _planner, robot_policy, _seed in trials if robot_policy}
        for scenario in sorted(robot_scenarios):
            has_robot = _scenario_has_robot(scenario)
            if has_robot is False:
                print(f"WARNING: scenario {scenario!r} has no kind=1 agent - robot_policy override will be a no-op and every robot metric will be NaN.")
            elif has_robot is None:
                print(f"WARNING: could not locate scenario yaml for {scenario!r}; cannot verify robot presence.")

    own = [t for i, t in enumerate(trials) if i % num_workers == worker_id]
    others = [t for i, t in enumerate(trials) if i % num_workers != worker_id]

    done_count = 0
    assigned = len(own)
    _write_progress(progress_file, done_count, assigned)

    overhead_csv = output_dir / "launch_overhead.csv"
    if not overhead_csv.exists():
        with overhead_csv.open("w") as f:
            f.write("scenario,planner,robot_policy,seed,total_wall_s,tick_wall_s,compute_s,overhead_s,overhead_pct\n")

    outcomes_csv = output_dir / "trial_outcomes.csv"
    if not outcomes_csv.exists():
        with outcomes_csv.open("w") as f:
            f.write("trial_id,attempt,rc,total_wall_s,integrity_ok,problems,bag_bytes,bag_action\n")

    pos = 0

    def run_one(trial: tuple[str, str, str, int], attempt: int = 1) -> bool:
        nonlocal pos
        pos += 1
        scenario, planner, robot_policy, seed = trial
        trial_id = _trial_dir_name(*trial)
        trial_dir = output_dir / trial_id
        sentinel = trial_dir / ".done"

        free = _free_bytes(output_dir)
        if free < MIN_FREE_BYTES:
            raise SystemExit(f"Disk guard: only {free / 1024**3:.1f} GB free under {output_dir} (need {MIN_FREE_BYTES / 1024**3:.0f} GB). Refusing to start {trial_id}.")

        if trial_dir.exists():
            manifest = trial_dir / "recording.yaml"
            if attempt > 1 and manifest.exists():
                keep_dir = output_dir / "_failed"
                keep_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(manifest, keep_dir / f"{trial_id}.attempt{attempt - 1}.recording.yaml")
            shutil.rmtree(trial_dir)

        rp = f", RobotPolicy={robot_policy}" if robot_policy else ""
        print("=========================================")
        print(f"Worker {worker_id} #{pos} (attempt {attempt}): Scenario={scenario}, Planner={planner}{rp}, Seed={seed}")
        print("=========================================")

        domain_id = worker_id * 2 + (pos % 2) + 2
        trial_env = {**os.environ, **_THREAD_ENV, "ROS_DOMAIN_ID": str(domain_id)}
        cmd = [
            "ros2",
            "launch",
            "arena_humansim",
            "arena_humansim.launch.py",
            f"scenario:={scenario}",
            f"local_planner:={planner}",
            f"force_local_planner:={'true' if scenario in force_planner_scenarios else 'false'}",
            f"seed:={seed}",
            "rtf:=0",
            f"time:={sim_duration}",
            "record:=True",
            f"record_dir:={trial_dir}",
            "render:=False",
            "rviz:=false",
            "markers:=0",
            "strict_recording:=true",
            f"trial_id:={trial_id}",
        ]
        if robot_policy:
            cmd.append(f"robot_policy:={robot_policy}")
        if robot_shutdown:
            cmd.append(f"robot_shutdown:={robot_shutdown}")
        if force_waypoint_mode:
            cmd.append(f"force_waypoint_mode:={force_waypoint_mode}")

        timeout = sim_duration * trial_timeout_factor if trial_timeout_factor > 0 else None
        print(f"Simulation started at max speed (ROS_DOMAIN_ID={domain_id}). Waiting for {sim_duration} sim-seconds to elapse...")
        total_wall, _sim_s, wall_s, compute_s, rc = _run_and_tee(cmd, trial_env, timeout=timeout)

        integrity_ok, problems = _integrity(trial_dir, rc)
        if integrity_ok:
            ok, why = _run_extractor(trial_dir)
            if not ok and not why.startswith("no extractor"):
                problems.append(why)
                integrity_ok = False
            elif not ok:
                print(f"  extraction skipped: {why}")

        bag_bytes = _bag_bytes(trial_dir)
        bag_action = "not pruned"
        if integrity_ok:
            sentinel.touch()
            print("Simulation closed cleanly.")
            if prune:
                bag_action = _prune_bag(trial_dir, trial_id, integrity_ok)
                print(f"  bag: {bag_action}")
        else:
            print(f"Trial failed: {trial_id}: {'; '.join(problems)}")

        with outcomes_csv.open("a", newline="") as f:
            csv.writer(f).writerow([trial_id, attempt, rc, f"{total_wall:.3f}", int(integrity_ok), "; ".join(problems), bag_bytes, bag_action])

        if wall_s is not None:
            overhead_s = total_wall - wall_s
            overhead_pct = 100.0 * overhead_s / total_wall if total_wall > 0 else 0.0
            print(f"  total={total_wall:.1f}s  tick_loop={wall_s:.1f}s  launch+teardown={overhead_s:.1f}s ({overhead_pct:.1f}%)")
            with overhead_csv.open("a") as f:
                f.write(f"{_scenario_id(scenario)},{planner},{robot_policy},{seed},{total_wall:.3f},{wall_s:.3f},{compute_s if compute_s is not None else ''},{overhead_s:.3f},{overhead_pct:.2f}\n")
        else:
            print(f"  total={total_wall:.1f}s  (final-rtf line not found; overhead unknown)")

        return integrity_ok

    def run_with_requeue(trial: tuple[str, str, str, int]) -> None:
        if run_one(trial, attempt=1):
            return
        print(f"Re-queuing {_trial_dir_name(*trial)} once.")
        run_one(trial, attempt=2)

    # Phase 1: own shard. Sibling-claimed and pre-existing-done trials drop out of `assigned`.
    for trial in own:
        name = _trial_dir_name(*trial)
        if (output_dir / name / ".done").exists():
            assigned -= 1
            _write_progress(progress_file, done_count, assigned)
            continue
        if not _try_claim(claims_dir, name):
            assigned -= 1
            _write_progress(progress_file, done_count, assigned)
            continue
        run_with_requeue(trial)
        done_count += 1
        _write_progress(progress_file, done_count, assigned)

    # Phase 2: steal anything siblings haven't claimed. Each successful claim grows `assigned`.
    for trial in others:
        name = _trial_dir_name(*trial)
        if (output_dir / name / ".done").exists():
            continue
        if not _try_claim(claims_dir, name):
            continue
        assigned += 1
        _write_progress(progress_file, done_count, assigned)
        run_with_requeue(trial)
        done_count += 1
        _write_progress(progress_file, done_count, assigned)

    print(f"Worker {worker_id} done: completed {done_count}/{assigned}.")
    print(f"Per-trial launch overhead: {overhead_csv}")
    print(f"Per-trial outcomes: {outcomes_csv}")


def _scenario_id(scenario: str) -> str:
    """Trial-safe scenario id: flatten path separators so trial-dir names don't contain slashes.
    Matches the YAML `name:` field for scenarios under config/evaluation/<bucket>/<density>/<modality>.yaml."""
    return scenario.replace("/", "_").replace("\\", "_").removesuffix(".yaml")


def _trial_dir_name(scenario: str, planner: str, robot_policy: str, seed: int) -> str:
    sid = _scenario_id(scenario)
    if robot_policy:
        return f"{sid}__{planner}__{robot_policy}__{seed}"
    return f"{sid}__{planner}__{seed}"


def _parse_trial_file(path: Path) -> list[tuple[str, str, str, int]]:
    trials: list[tuple[str, str, str, int]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) == 3:
            scenario, planner, seed = parts
            robot_policy = ""
        elif len(parts) == 4:
            scenario, planner, robot_policy, seed = parts
        else:
            raise ValueError(f"Bad trial line (expect 'scenario:planner:seed' or 'scenario:planner:robot_policy:seed'): {line!r}")
        trials.append((scenario, planner, robot_policy, int(seed)))
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Work-stealing sweep worker; iterates own modulo shard, then steals undone+unclaimed siblings.")
    parser.add_argument("--trial_file", type=str, required=True, help="Path to file with one 'scenario:planner:[robot_policy:]seed' per line. Same file for every worker.")
    parser.add_argument("--sim_duration", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True, help="<output_dir>/<trial_name>/ per trial; <output_dir>/.claims/<trial_name>.lock for atomic claims")
    parser.add_argument("--force_planner_scenarios", nargs="*", default=[])
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1, help="Total cooperating workers; this worker owns trials with index %% num_workers == worker_id.")
    parser.add_argument("--progress_file", type=str, default=None)
    parser.add_argument("--robot_shutdown", choices=("", "true", "false"), default="", help="end each trial when every robot reaches its goal; empty leaves the scenario value (default false).")
    parser.add_argument("--force_waypoint_mode", choices=WAYPOINT_MODES.values, default="", help="override scenario waypoint_mode for every kind=human agent (robots untouched). Empty = scenario value.")
    parser.add_argument("--trial_timeout_factor", type=float, default=3.0, help="kill a trial after factor x sim_duration seconds of wall clock (0 disables).")
    parser.add_argument("--prune", action="store_true", help="delete <trial>/bag after extraction unless the trial failed the integrity gate or falls in the deterministic 5%% keep sample.")
    args = parser.parse_args()

    trials = _parse_trial_file(Path(args.trial_file))
    run_sweep(
        trials=trials,
        sim_duration=args.sim_duration,
        output_dir=Path(args.output_dir).resolve(),
        force_planner_scenarios=set(args.force_planner_scenarios),
        worker_id=args.worker_id,
        num_workers=args.num_workers,
        progress_file=Path(args.progress_file).resolve() if args.progress_file else None,
        robot_shutdown=args.robot_shutdown,
        force_waypoint_mode=args.force_waypoint_mode,
        trial_timeout_factor=args.trial_timeout_factor,
        prune=args.prune,
    )


if __name__ == "__main__":
    main()
