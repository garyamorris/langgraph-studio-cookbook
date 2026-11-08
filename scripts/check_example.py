"""Check one or more examples against the quality contract without needing them registered in langgraph.json.

    .venv/bin/python scripts/check_example.py sql_analyst corrective_rag

Runs deepdive.common.testing.check_example (docs sections, META, offline end-to-end run, OpenAI strict-schema lint)
and prints the example's final output so you can eyeball it.
"""

from __future__ import annotations

import json
import os
import sys

os.environ["DEEPDIVE_OFFLINE"] = "1"

from deepdive.common.testing import check_example  # noqa: E402


def main(ids: list[str]) -> int:
    bad = 0
    for i in ids:
        try:
            res = check_example(f"deepdive.examples.{i}")
            print(f"ok   {i}  interrupts={len(res.interrupts)}")
            print("     output:", json.dumps(res.output, default=str)[:600])
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {i}: {type(e).__name__}: {e}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) if len(sys.argv) > 1 else print(__doc__) or 2)
