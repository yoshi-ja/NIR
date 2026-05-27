"""CLI entry point for the nir.akida toolchain.

Exposes four subcommands::

    python -m nir.akida.cli validate  my_model.nir --profile v1 --output-dir ./out
    python -m nir.akida.cli export    my_model.nir --profile v1 --output-dir ./out
    python -m nir.akida.cli quantize  my_model.nir --profile v1 --output-dir ./out
    python -m nir.akida.cli convert   my_model.nir --profile v1 --output-dir ./out
    python -m nir.akida.cli run       my_model.nir --profile v1 --output-dir ./out

Each stage writes artefacts and logs to ``--output-dir``.  When BrainChip
tools are absent the ``quantize`` and ``convert`` subcommands exit with a
clear installation message rather than a traceback.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import nir

from .profiles import AkidaTargetProfile
from .toolchain import (
    AkidaToolchainError,
    ConvertConfig,
    QuantizeConfig,
    ToolchainStage,
    convert_model,
    export_to_keras,
    run_akida_flow,
    validate_graph,
)
from .validator import AkidaValidator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _profile_from_str(s: str) -> AkidaTargetProfile:
    if s.lower() == "v1":
        return AkidaTargetProfile.v1()
    if s.lower() == "v2":
        return AkidaTargetProfile.v2()
    raise argparse.ArgumentTypeError(
        f"Unknown profile '{s}'.  Choose 'v1' or 'v2'."
    )


def _load_graph(path: str) -> nir.NIRGraph:
    try:
        return nir.read(path)
    except Exception as exc:
        print(f"ERROR: Could not read NIR graph from '{path}': {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    graph = _load_graph(args.nir_file)
    import pathlib
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = AkidaValidator(args.profile).validate(graph)
    report_dict = {
        "is_valid": report.is_valid,
        "summary": report.summary(),
        "diagnostics": [
            {
                "stage": d.stage.value,
                "node_name": d.node_name,
                "node_type": d.node_type,
                "message": d.message,
                "target_profile": d.target_profile,
            }
            for d in report.diagnostics
        ],
    }
    report_path = out / "validation_report.json"
    report_path.write_text(json.dumps(report_dict, indent=2))

    print(report.summary())
    for d in report.diagnostics:
        print(f"  {d}")
    print(f"\nReport written to: {report_path}")

    return 0 if report.is_valid else 1


def cmd_export(args: argparse.Namespace) -> int:
    graph = _load_graph(args.nir_file)
    import pathlib
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        path = export_to_keras(graph, args.profile, out)
        print(f"Keras model written to: {path}")
        return 0
    except (AkidaToolchainError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    graph = _load_graph(args.nir_file)

    stop_map = {
        "validation": ToolchainStage.VALIDATION,
        "export": ToolchainStage.EXPORT,
        "quantization": ToolchainStage.QUANTIZATION,
        None: None,
    }
    stop_after = stop_map.get(args.stop_after)

    try:
        result = run_akida_flow(
            graph=graph,
            profile=args.profile,
            output_dir=args.output_dir,
            quantize_config=QuantizeConfig(num_samples=args.num_samples),
            stop_after=stop_after,
        )
        print(result)
        return 0
    except AkidaToolchainError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(
            f"ERROR: A required dependency is not installed.\n{exc}",
            file=sys.stderr,
        )
        return 1


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nir.akida.cli",
        description="nir.akida — BrainChip Akida deployment toolchain for NIR graphs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("nir_file", help="Path to the NIR graph file (.nir / .h5)")
    common.add_argument(
        "--profile",
        type=_profile_from_str,
        default="v1",
        metavar="{v1,v2}",
        help="Akida hardware target profile (default: v1)",
    )
    common.add_argument(
        "--output-dir",
        default="./akida_output",
        metavar="DIR",
        help="Directory for artefacts and logs (default: ./akida_output)",
    )

    # validate
    p = sub.add_parser("validate", parents=[common], help="Validate a NIR graph for Akida compatibility")
    p.set_defaults(func=cmd_validate)

    # export
    p = sub.add_parser("export", parents=[common], help="Export a validated NIR graph to Keras")
    p.set_defaults(func=cmd_export)

    # run (full pipeline)
    p = sub.add_parser("run", parents=[common], help="Run the full NIR → Akida pipeline")
    p.add_argument(
        "--stop-after",
        choices=["validation", "export", "quantization"],
        default=None,
        help="Stop the pipeline after a specific stage (default: run all stages)",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=1024,
        metavar="N",
        help="Number of calibration samples for QuantizeML (default: 1024)",
    )
    p.set_defaults(func=cmd_run)

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
