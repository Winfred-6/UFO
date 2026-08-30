#!/usr/bin/env python3
"""Ask a UFO training run to checkpoint its latest complete update and exit."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from humanoidverse.training.safe_stop import read_json_if_present, request_safe_stop, safe_stop_status_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="Work directory of the running training job.")
    parser.add_argument("--reason", default="user request", help="Reason recorded in the safe-stop status file.")
    parser.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait until training confirms that the final checkpoint was written.",
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="Maximum seconds to wait for checkpoint confirmation.")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> None:
    args = parse_args()
    request = request_safe_stop(args.work_dir, reason=args.reason)
    work_dir = args.work_dir.expanduser().resolve()
    print(
        f"Safe-stop request {request['request_id']} written for {work_dir}. "
        "Training will stop after saving its next complete update boundary.",
        flush=True,
    )
    if not args.wait:
        return

    deadline = time.monotonic() + args.timeout
    status_path = safe_stop_status_path(work_dir)
    while time.monotonic() < deadline:
        status = read_json_if_present(status_path)
        if status is not None and status.get("request_id") == request["request_id"]:
            if status.get("status") != "saved_and_stopped":
                raise RuntimeError(f"Training returned an unexpected safe-stop status: {status}")
            print(
                "Safe stop complete: "
                f"global_time={status['global_time']}, local_time={status['local_time']}, "
                f"optimizer_steps={status['optimizer_steps']}, checkpoint={status['checkpoint_dir']}",
                flush=True,
            )
            return
        time.sleep(0.5)

    raise TimeoutError(
        f"Timed out after {args.timeout:g}s waiting for {status_path}. "
        "The request file was left in place, so a running trainer can still process it."
    )


if __name__ == "__main__":
    main()
