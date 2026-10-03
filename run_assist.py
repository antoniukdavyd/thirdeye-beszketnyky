#!/usr/bin/env python3
"""Entrypoint: python run_assist.py"""

from __future__ import annotations

import argparse

from assist.app import AssistApp


def main():
    parser = argparse.ArgumentParser(description="nekit blind-assist MVP (Mac + Record3D)")
    parser.add_argument("--device", type=int, default=0, help="Record3D device index")
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO-World")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice session")
    args = parser.parse_args()

    app = AssistApp(enable_yolo=not args.no_yolo, enable_voice=not args.no_voice)
    app.connect(dev_idx=args.device)
    app.run()


if __name__ == "__main__":
    main()
