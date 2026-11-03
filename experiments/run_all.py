"""The single entry point for the experiments.

    python experiments/run_all.py                 # everything, resumes where it stopped
    python experiments/run_all.py --list          # what would run
    python experiments/run_all.py --group main    # one group
    python experiments/run_all.py --only t2_single_s42 t2_multitask_s42
    python experiments/run_all.py --no-analyze    # train only

Finished runs are skipped, so it is safe to interrupt and restart.
"""

from __future__ import annotations

import argparse
import sys
import time

import jobs
import run_experiment as R


def select(groups: list[str] | None, names: list[str] | None) -> list[dict]:
    if names:
        by_name = {job["name"]: job for job in jobs.ALL}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise SystemExit(f"unknown run(s): {missing}")
        return [by_name[name] for name in names]
    if groups:
        unknown = [group for group in groups if group not in jobs.GROUPS]
        if unknown:
            raise SystemExit(f"unknown group(s): {unknown}; choose from {list(jobs.GROUPS)}")
        return [job for group in groups for job in jobs.GROUPS[group]]
    return list(jobs.ALL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", nargs="*", default=None, help=f"one of {list(jobs.GROUPS)}")
    parser.add_argument("--only", nargs="*", default=None, help="explicit run names")
    parser.add_argument("--list", action="store_true", help="print the selection and exit")
    parser.add_argument("--no-analyze", action="store_true", help="do not regenerate the paper artefacts")
    args = parser.parse_args()

    selected = select(args.group, args.only)
    if args.list:
        for job in selected:
            print(job["name"])
        print(f"\n{len(selected)} run(s)")
        return 0

    started = time.time()
    print(f"{len(selected)} run(s) selected", flush=True)
    for index, job in enumerate(selected, 1):
        config = R.Config(**job)
        print(f"\n[{index}/{len(selected)}] {config.name}", flush=True)
        summary = R.run(config)
        print(
            f"[{index}/{len(selected)}] {config.name}: macro_f1={summary['test_macro_f1']:.4f}",
            flush=True,
        )

    if not args.no_analyze:
        print("\nregenerating tables, figures and numbers", flush=True)
        from analyze_results import main as analyze

        analyze()

    print(f"\nfinished in {(time.time() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
