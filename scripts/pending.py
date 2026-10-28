"""Explicit placeholders; never report fake results."""
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("command", choices=["train", "eval", "demo"])
parser.add_argument("--fast", action="store_true", help="Reserved for deterministic subsampling")
args = parser.parse_args()
parser.exit(2, f"{args.command} is not implemented in Phase 0. Review PLAN.md first.\n")
