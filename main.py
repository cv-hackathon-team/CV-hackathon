"""DepthGuard360 command-line entry point: train, evaluate or check datasets."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse the subcommand and the ``--config`` path."""
    # TODO: add train, evaluate and check-datasets subcommands.
    raise NotImplementedError


def main() -> None:
    """Load the config and dispatch to the requested subcommand."""
    # TODO: implement.
    raise NotImplementedError


if __name__ == "__main__":
    main()
