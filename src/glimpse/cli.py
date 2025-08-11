# src/glimpse/cli.py
import argparse

def main():
    parser = argparse.ArgumentParser(description="Glimpse CLI")
    sub = parser.add_subparsers(dest="command")

    trace_cmd = sub.add_parser("trace", help="Trace a script or function")
    trace_cmd.add_argument("--path", help="Path to file to trace")

    args = parser.parse_args()

    if args.command == "trace":
        print(f"Tracing file: {args.path}")
