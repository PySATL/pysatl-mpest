#!/usr/bin/env python3
"""Export every page of every ``.drawio`` file to a PNG next to it.

The PNG of a page is named ``<file stem>.<page id>.png`` and is placed in the same ``diagrams/`` directory as
the source. A PNG that is newer than its ``.drawio`` is skipped unless ``--force`` is given.

Usage::

    python3 docs/tools/export_diagrams.py [--force] [--scale 2] [PATH ...]

``PATH`` is a ``.drawio`` file or a directory searched recursively.
The default is ``docs/source/design``.
Needs the draw.io desktop CLI 27.0.2 or newer: page numbers are 1-based
(``drawio`` on ``PATH`` or ``/opt/drawio/drawio``).
"""

import argparse
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "source" / "design"
DRAWIO_CANDIDATES = ("drawio", "draw.io", "/opt/drawio/drawio")


def find_drawio() -> str:
    for name in DRAWIO_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    sys.exit("error: draw.io CLI not found (tried drawio, draw.io, /opt/drawio/drawio)")


def collect(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        files += [path] if path.is_file() else sorted(path.rglob("*.drawio"))
    return files


def export_file(drawio: str, source: Path, scale: float, force: bool) -> int:
    """Export the pages of one file; returns the number of exported PNGs."""
    exported = 0
    pages = ET.parse(source).getroot().findall("diagram")
    seen: set[str] = set()
    for index, page in enumerate(pages):
        page_id = page.get("id")
        if not page_id:
            sys.exit(f"error: {source}: page {index + 1} has no id")
        if page_id in seen:
            sys.exit(f"error: {source}: duplicate page id {page_id!r}")
        seen.add(page_id)
        target = source.with_name(f"{source.stem}.{page_id}.png")
        if not force and target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
            continue
        cmd = [
            drawio,
            "-x",
            "-f",
            "png",
            "-p",
            str(index + 1),
            "-s",
            str(scale),
            "-b",
            "10",
            "--theme",
            "light",
            "-o",
            str(target),
            str(source),
            "--no-sandbox",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
        if not target.exists():
            sys.exit(f"error: export of {source} page {page_id!r} failed:\n{result.stderr[-500:]}")
        print(f"{source.name} [{page_id}] -> {target.name}")
        exported += 1
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, default=[DEFAULT_ROOT])
    parser.add_argument("--force", action="store_true", help="re-export even if the PNG is up to date")
    parser.add_argument("--scale", type=float, default=2.0)
    args = parser.parse_args()
    drawio = find_drawio()
    total = sum(export_file(drawio, f, args.scale, args.force) for f in collect(args.paths))
    print(f"exported {total} PNG")


if __name__ == "__main__":
    main()
