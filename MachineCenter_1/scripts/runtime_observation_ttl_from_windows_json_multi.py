from __future__ import annotations

"""
scripts/runtime_observation_ttl_from_windows_json_multi.py

Convert CNC MultiSensorWindow runtime JSON files into one TTL file.

This version is updated for runtime JSON files containing:

    sth_status.battery_level
    sth_status.comm_score
    sth_status.max_torque

It also supports:
    1. A single runtime window object
    2. A list of runtime window objects
    3. An object with a "windows" list
    4. JSON files that accidentally contain trailing commas

Run from the PROJECT ROOT:

    # Default: read all *_runtime_*.json and *.json files in samples/
    python scripts/runtime_observation_ttl_from_windows_json_multi.py

    # Explicit files only
    python scripts/runtime_observation_ttl_from_windows_json_multi.py ^
        --json samples/_runtime_w10233.json samples/_runtime_w10234.json ^
        --json-dir "" ^
        --out runtime/_runtime_observation.ttl

    # Explicit directory
    python scripts/runtime_observation_ttl_from_windows_json_multi.py ^
        --json-dir samples ^
        --out runtime/_runtime_observation.ttl
"""

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any, Iterable

from rdflib import Graph, Literal, Namespace, RDF, RDFS
from rdflib.namespace import XSD

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path.cwd()

DEFAULT_JSON_DIR = "samples"
DEFAULT_OUTPUT_TTL = "runtime/CNC_runtime_observation.ttl"

# ---------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------
MIS = Namespace("http://nkust.edu.tw/mislab#")
STH = Namespace("http://nkust.edu.tw/mislab/cnc/ontology/sth#")

# ---------------------------------------------------------------------
# Required runtime fields
# ---------------------------------------------------------------------
REQUIRED_FIELDS = (
    "window_id",
    "project",
    "batch",
    "SmartToolHolder",
    "timestamp",
    "time_window.start",
    "time_window.end",
    "alignment.nc_block",
    "alignment.tool_path_segment",
    "process.spindle_speed",
    "process.feed_rate",
    "features.torque_ratio",
    "features.vibration_rms",
    "features.chatter_score",
    "features.measured_ra",
    "sth_status.battery_level",
    "sth_status.comm_score",
    "sth_status.max_torque",
    "progress.sequence_index",
    "progress.progress_ratio",
    "progress.block_progress_ratio",
)


# ---------------------------------------------------------------------
# Path and JSON helpers
# ---------------------------------------------------------------------
def resolve_path(path_like: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def expand_json_inputs(
    json_inputs: Iterable[str] | None = None,
    json_dir: str | None = DEFAULT_JSON_DIR,
) -> list[Path]:
    """
    Resolve explicit JSON paths, glob patterns, and/or a JSON directory into files.

    Default behavior:
        json_inputs=None and json_dir="samples" means read JSON files in samples/.
    """
    files: list[Path] = []

    if json_inputs:
        for item in json_inputs:
            pattern = item if Path(item).is_absolute() else str(PROJECT_ROOT / item)
            matches = [Path(p) for p in glob.glob(pattern)]
            if matches:
                files.extend(matches)
            else:
                files.append(resolve_path(item))

    if json_dir:
        directory = resolve_path(json_dir)
        if not directory.exists():
            raise FileNotFoundError(f"JSON directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"--json-dir is not a directory: {directory}")

        # Prefer runtime files, but also include normal JSON files.
        files.extend(directory.glob("*_runtime_*.json"))
        files.extend(directory.glob("*.json"))

    unique_files = sorted({p.resolve() for p in files})

    if not unique_files:
        raise ValueError(
            "No JSON input files were found. "
            "Put JSON files in samples/ or provide --json/--json-dir."
        )

    missing = [p for p in unique_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "JSON file(s) not found:\n  " + "\n  ".join(str(p) for p in missing)
        )

    non_json = [p for p in unique_files if p.suffix.lower() != ".json"]
    if non_json:
        raise ValueError(
            "Input file(s) must have .json extension:\n  " + "\n  ".join(str(p) for p in non_json)
        )

    return unique_files


def remove_trailing_commas(text: str) -> str:
    """
    Remove trailing commas before } or ].

    This is useful for files such as:
        "measured_ra": 1.12,
      }
    """
    return re.sub(r",\s*([}\]])", r"\1", text)


def load_json_text_with_repair(json_file: Path) -> Any:
    """
    Load JSON. If strict JSON parsing fails, retry after removing trailing commas.
    """
    text = json_file.read_text(encoding="utf-8")

    try:
        return json.loads(text)
    except json.JSONDecodeError as first_exc:
        repaired = remove_trailing_commas(text)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON file after repair attempt: {json_file}\n{first_exc}") from first_exc


def require(data: dict[str, Any], path: str, source: Path | None = None) -> Any:
    cur: Any = data

    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            src = f" in {source}" if source else ""
            raise ValueError(f"Missing required JSON field: {path}{src}")
        cur = cur[key]

    if cur is None or cur == "":
        src = f" in {source}" if source else ""
        raise ValueError(f"Empty value is not allowed for JSON field: {path}{src}")

    return cur


def optional(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data

    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]

    return default if cur is None or cur == "" else cur


def safe_local_name(value: Any) -> str:
    """
    Convert IDs such as STH-2 into safe URI local names: STH_2.
    Existing underscores are preserved.
    """
    text = str(value).strip()
    text = text.replace("-", "_")
    text = re.sub(r"[^A-Za-z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


# ---------------------------------------------------------------------
# RDF literal helpers
# ---------------------------------------------------------------------
def add_string(g: Graph, s, p, value: Any) -> None:
    g.add((s, p, Literal(str(value), datatype=XSD.string)))


def add_datetime(g: Graph, s, p, value: Any) -> None:
    g.add((s, p, Literal(str(value), datatype=XSD.dateTime)))


def add_double(g: Graph, s, p, value: Any, field_name: str, source: Path | None = None) -> None:
    try:
        g.add((s, p, Literal(float(value), datatype=XSD.double)))
    except (TypeError, ValueError) as exc:
        src = f" in {source}" if source else ""
        raise ValueError(f"JSON field '{field_name}' must be numeric, got: {value!r}{src}") from exc


def add_integer(g: Graph, s, p, value: Any, field_name: str, source: Path | None = None) -> None:
    try:
        g.add((s, p, Literal(int(value), datatype=XSD.integer)))
    except (TypeError, ValueError) as exc:
        src = f" in {source}" if source else ""
        raise ValueError(f"JSON field '{field_name}' must be integer, got: {value!r}{src}") from exc


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
def validate_required_fields(data: dict[str, Any], source: Path | None = None) -> None:
    for field in REQUIRED_FIELDS:
        require(data, field, source)


# ---------------------------------------------------------------------
# RDF graph construction
# ---------------------------------------------------------------------
def add_identity_nodes(
    g: Graph,
    project_id: str,
    batch_id: str,
    holder_id: str,
    nc_block_id: str,
    segment_id: str,
):
    project_uri = MIS[safe_local_name(project_id)]
    batch_uri = MIS[safe_local_name(batch_id)]
    holder_uri = STH[safe_local_name(holder_id)]
    nc_block_uri = MIS[safe_local_name(nc_block_id)]
    segment_uri = MIS[safe_local_name(segment_id)]

    g.add((project_uri, RDF.type, MIS.Project))
    g.add((project_uri, RDFS.label, Literal(project_id, datatype=XSD.string)))

    g.add((batch_uri, RDF.type, MIS.Batch))
    g.add((batch_uri, RDFS.label, Literal(batch_id, datatype=XSD.string)))

    g.add((holder_uri, RDF.type, STH.SmartToolHolder))
    g.add((holder_uri, RDFS.label, Literal(holder_id, datatype=XSD.string)))

    g.add((nc_block_uri, RDF.type, STH.NCBlock))
    g.add((nc_block_uri, RDFS.label, Literal(nc_block_id, datatype=XSD.string)))

    g.add((segment_uri, RDF.type, STH.ToolPathSegment))
    g.add((segment_uri, RDFS.label, Literal(segment_id, datatype=XSD.string)))

    return project_uri, batch_uri, holder_uri, nc_block_uri, segment_uri


def build_one_window(data: dict[str, Any], g: Graph, source: Path | None = None) -> str:
    validate_required_fields(data, source)

    window_id = str(require(data, "window_id", source))
    project_id = str(require(data, "project", source))
    batch_id = str(require(data, "batch", source))
    holder_id = str(require(data, "SmartToolHolder", source))
    nc_block_id = str(require(data, "alignment.nc_block", source))
    segment_id = str(require(data, "alignment.tool_path_segment", source))

    window_uri = MIS[f"RuntimeWindow_{safe_local_name(window_id)}"]

    project_uri, batch_uri, holder_uri, nc_block_uri, segment_uri = add_identity_nodes(
        g=g,
        project_id=project_id,
        batch_id=batch_id,
        holder_id=holder_id,
        nc_block_id=nc_block_id,
        segment_id=segment_id,
    )

    # Main runtime window
    g.add((window_uri, RDF.type, STH.MultiSensorWindow))
    add_string(g, window_uri, MIS.hasWindowId, window_id)

    # Context relations
    g.add((window_uri, MIS.belongsToProject, project_uri))
    g.add((window_uri, MIS.belongsToBatch, batch_uri))

    # Add both property names for compatibility with existing rules/shapes.
    g.add((window_uri, MIS.hasSmartToolHolder, holder_uri))
    g.add((window_uri, STH.usesSmartToolHolder, holder_uri))

    add_datetime(g, window_uri, MIS.hasTimestamp, require(data, "timestamp", source))
    add_datetime(g, window_uri, STH.hasStartTime, require(data, "time_window.start", source))
    add_datetime(g, window_uri, STH.hasEndTime, require(data, "time_window.end", source))

    g.add((window_uri, STH.alignedWithNCBlock, nc_block_uri))
    g.add((window_uri, STH.alignedWithToolPathSegment, segment_uri))

    # Process values
    add_double(g, window_uri, MIS.hasSpindleSpeed, require(data, "process.spindle_speed", source), "process.spindle_speed", source)
    add_double(g, window_uri, MIS.hasFeedRate, require(data, "process.feed_rate", source), "process.feed_rate", source)

    # Runtime machining features
    add_double(g, window_uri, MIS.hasTorqueRatio, require(data, "features.torque_ratio", source), "features.torque_ratio", source)
    add_double(g, window_uri, MIS.hasVibrationRMS, require(data, "features.vibration_rms", source), "features.vibration_rms", source)

    acoustic_rms = optional(data, "features.acoustic_rms")
    if acoustic_rms is not None:
        add_double(g, window_uri, MIS.hasAcousticRMS, acoustic_rms, "features.acoustic_rms", source)

    add_double(g, window_uri, MIS.hasChatterScoreObserved, require(data, "features.chatter_score", source), "features.chatter_score", source)
    # Also add the simpler name for compatibility with bindings/shapes that use mis:hasChatterScore.
    add_double(g, window_uri, MIS.hasChatterScore, require(data, "features.chatter_score", source), "features.chatter_score", source)

    add_double(g, window_uri, MIS.hasMeasuredRa, require(data, "features.measured_ra", source), "features.measured_ra", source)

    # Smart-tool-holder status values.
    # These are required by STH threshold knowledge rules.
    add_double(g, window_uri, MIS.hasBatteryLevel, require(data, "sth_status.battery_level", source), "sth_status.battery_level", source)
    add_double(g, window_uri, MIS.hasCommScore, require(data, "sth_status.comm_score", source), "sth_status.comm_score", source)
    add_double(g, window_uri, MIS.hasMaxTorque, require(data, "sth_status.max_torque", source), "sth_status.max_torque", source)

    # Progress values
    add_integer(g, window_uri, MIS.hasSequenceIndex, require(data, "progress.sequence_index", source), "progress.sequence_index", source)
    add_double(g, window_uri, MIS.hasProgressRatio, require(data, "progress.progress_ratio", source), "progress.progress_ratio", source)
    add_double(g, window_uri, MIS.hasBlockProgressRatio, require(data, "progress.block_progress_ratio", source), "progress.block_progress_ratio", source)

    return window_id


def load_json_payload(json_file: Path) -> list[dict[str, Any]]:
    data = load_json_text_with_repair(json_file)

    if isinstance(data, list):
        windows = data
    elif isinstance(data, dict) and isinstance(data.get("windows"), list):
        windows = data["windows"]
    elif isinstance(data, dict):
        windows = [data]
    else:
        raise ValueError(
            f"Runtime JSON must be an object, a list of objects, "
            f"or an object with a 'windows' list: {json_file}"
        )

    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            raise ValueError(f"Window entry #{index} in {json_file} must be a JSON object.")

    return windows


def build_graph_from_json_files(json_files: list[Path]) -> tuple[Graph, list[str]]:
    g = Graph()
    g.bind("mis", MIS)
    g.bind("sth", STH)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    window_ids: list[str] = []
    seen_ids: set[str] = set()

    for json_file in json_files:
        windows = load_json_payload(json_file)

        for window in windows:
            window_id = str(require(window, "window_id", json_file))

            if window_id in seen_ids:
                raise ValueError(f"Duplicate window_id '{window_id}' found while reading {json_file}")

            seen_ids.add(window_id)
            build_one_window(window, g, json_file)
            window_ids.append(window_id)

    return g, window_ids


def convert_json_to_ttl(json_files: list[Path], output_ttl: str | Path) -> Graph:
    output_ttl = resolve_path(output_ttl)
    g, window_ids = build_graph_from_json_files(json_files)

    output_ttl.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=output_ttl, format="turtle")

    print("[OK] Generated runtime TTL")
    print(f"     Project root: {PROJECT_ROOT}")
    print("     JSON files:")
    for json_file in json_files:
        print(f"       - {json_file}")
    print(f"     TTL:     {output_ttl}")
    print(f"     Windows: {len(window_ids)} ({', '.join(window_ids)})")
    print(f"     Triples: {len(g)}")

    return g


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CNC runtime JSON files into one MultiSensorWindow TTL."
    )
    parser.add_argument(
        "--json",
        nargs="+",
        default=None,
        help=(
            "Optional explicit JSON files or glob patterns. "
            "Example: --json samples/_runtime_w*.json"
        ),
    )
    parser.add_argument(
        "--json-dir",
        default=DEFAULT_JSON_DIR,
        help=(
            "Directory containing JSON files. Default: samples. "
            "Use --json-dir '' to disable directory scanning when using --json only."
        ),
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT_TTL,
        help=f"Output TTL path. Default: {DEFAULT_OUTPUT_TTL}",
    )

    args = parser.parse_args()

    # Allow --json-dir "" to mean no directory scan.
    json_dir = args.json_dir if args.json_dir else None

    json_files = expand_json_inputs(args.json, json_dir)
    convert_json_to_ttl(json_files, args.out)


if __name__ == "__main__":
    main()
