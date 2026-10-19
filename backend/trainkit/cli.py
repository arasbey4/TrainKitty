"""TrainKit backend launcher CLI (playwright of the loopback bind)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="trainkit-server")
    parser.add_argument("--host", default="127.0.0.1", help="(loopback by default)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    from .api.app import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
