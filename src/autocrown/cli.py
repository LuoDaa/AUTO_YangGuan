from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .frame import build_confirmed_frame
from .intaglio import build_gap_surface_preview
from .meshops import load_mesh, save_mesh
from .pipeline import build_outer_crown


def _load_margin(path: str | Path) -> np.ndarray:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("points") or data.get("margin")
    margin = np.asarray(data, float)
    if margin.ndim != 2 or margin.shape[1] != 3:
        raise ValueError("margin.json must contain Nx3 points")
    return margin


def _load_frame(path: str | Path, margin: np.ndarray):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return build_confirmed_frame(
        margin,
        data["mesial"],
        data["buccal"],
        data["insertion"],
    )


def _outer(args) -> int:
    margin = _load_margin(args.margin)
    frame = _load_frame(args.frame, margin)
    template = load_mesh(args.template)
    mesial = load_mesh(args.mesial) if args.mesial else None
    distal = load_mesh(args.distal) if args.distal else None
    antagonist = load_mesh(args.antagonist) if args.antagonist else None
    _mesh, report = build_outer_crown(
        template,
        margin,
        frame,
        mesial=mesial,
        distal=distal,
        antagonist=antagonist,
        config=PipelineConfig(),
        out_dir=args.out,
    )
    print(json.dumps({"status": report["status"], "artifacts": report["artifacts"]},
                     ensure_ascii=False, indent=2))
    return 0


def _intaglio_preview(args) -> int:
    margin = _load_margin(args.margin)
    frame = _load_frame(args.frame, margin)
    support = load_mesh(args.support)
    preview, report = build_gap_surface_preview(
        support, margin, frame, cement_gap_mm=args.gap,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_mesh(preview, out)
    report_path = out.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "mesh": str(out), "report": str(report_path)},
                     ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autocrown")
    sub = parser.add_subparsers(dest="command", required=True)

    outer = sub.add_parser("outer", help="build conservative outer crown candidate")
    outer.add_argument("--template", required=True)
    outer.add_argument("--margin", required=True)
    outer.add_argument("--frame", required=True)
    outer.add_argument("--mesial")
    outer.add_argument("--distal")
    outer.add_argument("--antagonist")
    outer.add_argument("--out", required=True, help="output directory")
    outer.set_defaults(func=_outer)

    inner = sub.add_parser("intaglio-preview", help="experimental support gap-surface preview")
    inner.add_argument("--support", required=True)
    inner.add_argument("--margin", required=True)
    inner.add_argument("--frame", required=True)
    inner.add_argument("--gap", type=float, default=0.06)
    inner.add_argument("--out", required=True, help="output STL path")
    inner.set_defaults(func=_intaglio_preview)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
