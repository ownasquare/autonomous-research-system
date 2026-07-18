"""Wait for a local HTTP health endpoint without shell polling."""

from __future__ import annotations

import argparse
import time

import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            response = httpx.get(args.url, timeout=2.0)
            if response.is_success:
                return 0
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
