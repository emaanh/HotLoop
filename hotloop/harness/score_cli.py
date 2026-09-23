"""Score a solution read from stdin (used by non-Modal backends inside their scoring container).

    python -m hotloop.harness.score_cli TASK_ID [--public] < solution.py
Prints the result JSON on a line starting with HOTLOOP_RESULT.
"""

import argparse
import json
import os
import sys

from hotloop.bench.store import Store
from hotloop.harness.service import run_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_id")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()
    store = Store.default()
    if os.getuid() == 0 and os.path.isdir(store.hidden):
        os.chmod(store.hidden, 0o700)  # the solution runner drops to an unprivileged user
    result = run_eval(store, args.task_id, sys.stdin.read(), hidden=not args.public)
    print("HOTLOOP_RESULT " + json.dumps(result, default=str))


if __name__ == "__main__":
    main()
