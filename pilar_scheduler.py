"""
Run PILAR scheduled jobs in a dedicated process.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("PILAR_ENABLE_SCHEDULER", "1")

import app


def main() -> int:
    status = app.get_scheduler_status()
    if not status.get("started"):
        print(f"[Pilar/scheduler] Scheduler not running: {status.get('reason')}")
        return 1

    print(
        "[Pilar/scheduler] Scheduler active "
        f"({status.get('lock_kind')} leader). Press Ctrl-C to stop."
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("[Pilar/scheduler] Stopping")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
