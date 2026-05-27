from __future__ import annotations

"""
validate_runtime_layer_corrected.py

Purpose
-------
Validate only real CNC/TMV720 runtime MultiSensorWindow instances against
CNC_generated_shapes.ttl.

This corrected version is aligned with the uploaded files:

    runtime/CNC_runtime_observation.ttl
    shapes/CNC_generated_shapes.ttl

Main corrections
----------------
1. Keeps only runtime windows whose URI starts with mis:RuntimeWindow_.
2. Also keeps the rdf:type/rdfs:label triples of referenced object nodes
   such as Project, Batch, NCBlock, ToolPathSegment, and SmartToolHolder.
   This is necessary for sh:class validation.
3. Patches shape paths that use unit-bearing property IRIs, for example:

       mis:hasSpindleSpeed(rpm/min)  -> mis:hasSpindleSpeed
       mis:hasFeedRate(mm/min)       -> mis:hasFeedRate
       mis:hasMeasuredRa(um)         -> mis:hasMeasuredRa

   because the uploaded runtime TTL uses the clean property names.
4. Supports command-line arguments and safe path resolution from either
   the project root or validation/ folder.

Run from project root:

    python validation/validate_runtime_layer_corrected.py

or explicitly:

    python validation/validate_runtime_layer_corrected.py \
        --runtime runtime/CNC_runtime_observation.ttl \
        --shapes shapes/CNC_generated_shapes.ttl \
        --report-txt output/runtime_validation_report.txt \
        --report-ttl output/runtime_validation_report.ttl
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable

try:
    from pyshacl import validate
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "[ERROR] pySHACL is not installed. Install it with:\n"
        "        pip install pyshacl rdflib\n"
    ) from exc

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import XSD


# -----------------------------------------------------------------------------
# Namespaces
# -----------------------------------------------------------------------------
MIS = Namespace("http://nkust.edu.tw/mislab#")
STH = Namespace("http://nkust.edu.tw/mislab/cnc/ontology/sth#")
SH = Namespace("http://www.w3.org/ns/shacl#")


# -----------------------------------------------------------------------------
# Project path helpers
# -----------------------------------------------------------------------------
def infer_project_root() -> Path:
    """
    If this script is located in validation/, the project root is its parent.
    If this script is located directly in the project root, the project root is
    the script directory.
    """
    here = Path(__file__).resolve().parent
    if here.name.lower() in {"validation", "scripts"}:
        return here.parent
    return here


PROJECT_ROOT = infer_project_root()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


# -----------------------------------------------------------------------------
# Graph loading
# -----------------------------------------------------------------------------
def bind_common_prefixes(g: Graph) -> None:
    g.bind("mis", MIS)
    g.bind("sth", STH)
    g.bind("sh", SH)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)


def parse_required_ttl(path: Path, label: str) -> Graph:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    if path.suffix.lower() not in {".ttl", ".rdf", ".owl", ".nt", ".n3"}:
        raise ValueError(f"{label} file should be an RDF/Turtle file: {path}")

    g = Graph()
    bind_common_prefixes(g)
    print(f"[LOAD] {label}: {path}")
    g.parse(path, format="turtle")
    return g


# -----------------------------------------------------------------------------
# Runtime graph extraction
# -----------------------------------------------------------------------------
def is_runtime_window(node: URIRef) -> bool:
    return str(node).startswith(str(MIS) + "RuntimeWindow_")


def copy_descriptive_triples(raw_g: Graph, out_g: Graph, node: URIRef) -> None:
    """
    Copy minimal object metadata needed by SHACL sh:class validation and readable
    reports. Avoid copying arbitrary unrelated static ontology/demo data.
    """
    for p, o in raw_g.predicate_objects(node):
        if p in {RDF.type, RDFS.label}:
            out_g.add((node, p, o))


def keep_only_runtime_layer(raw_g: Graph) -> Graph:
    """
    Keep only runtime MultiSensorWindow nodes and directly referenced object
    descriptions.

    The old script copied only predicate-object triples of the runtime window.
    That caused sh:class failures because objects such as mis:Batch_A or
    mis:2026SBIR had no rdf:type in the validation data graph.
    """
    out_g = Graph()
    bind_common_prefixes(out_g)

    runtime_windows = [
        s for s in raw_g.subjects(RDF.type, STH.MultiSensorWindow)
        if isinstance(s, URIRef) and is_runtime_window(s)
    ]

    for window in runtime_windows:
        for p, o in raw_g.predicate_objects(window):
            out_g.add((window, p, o))

            if isinstance(o, URIRef):
                copy_descriptive_triples(raw_g, out_g, o)

    return out_g


# -----------------------------------------------------------------------------
# Shape graph correction for uploaded runtime vocabulary
# -----------------------------------------------------------------------------
PROPERTY_PATH_ALIASES: dict[URIRef, URIRef] = {
    MIS["hasSpindleSpeed(rpm/min)"]: MIS.hasSpindleSpeed,
    MIS["hasFeedRate(mm/min)"]: MIS.hasFeedRate,
    MIS["hasMeasuredRa(um)"]: MIS.hasMeasuredRa,
}

# Optional aliases. They do not hurt if the shape does not contain them.
OPTIONAL_PROPERTY_PATH_ALIASES: dict[URIRef, URIRef] = {
    MIS.hasChatterScoreObserved: MIS.hasChatterScore,
}


def patch_shape_paths(shapes_g: Graph) -> int:
    """
    Replace sh:path values in the shapes graph when the shape uses older or
    unit-bearing property IRIs but the runtime TTL uses clean canonical property
    names.
    """
    replacements = {**PROPERTY_PATH_ALIASES, **OPTIONAL_PROPERTY_PATH_ALIASES}
    changed = 0

    for shape_node, _, old_path in list(shapes_g.triples((None, SH.path, None))):
        if old_path in replacements:
            new_path = replacements[old_path]
            shapes_g.remove((shape_node, SH.path, old_path))
            shapes_g.add((shape_node, SH.path, new_path))
            changed += 1
            print(f"[PATCH] sh:path {old_path} -> {new_path}")

    return changed


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
def qname(g: Graph, node) -> str:
    if node is None:
        return "None"
    try:
        return g.namespace_manager.normalizeUri(node)
    except Exception:
        return str(node)


def summarize_runtime(data_g: Graph) -> None:
    windows = sorted(set(data_g.subjects(RDF.type, STH.MultiSensorWindow)), key=str)

    print("\n===== Runtime Layer Summary =====")
    print(f"Runtime triples:       {len(data_g)}")
    print(f"Runtime windows:       {len(windows)}")

    for w in windows:
        holder = data_g.value(w, MIS.hasSmartToolHolder) or data_g.value(w, STH.usesSmartToolHolder)
        project = data_g.value(w, MIS.belongsToProject)
        batch = data_g.value(w, MIS.belongsToBatch)
        print(f"  - {qname(data_g, w)}")
        print(f"      Project:         {qname(data_g, project)}")
        print(f"      Batch:           {qname(data_g, batch)}")
        print(f"      SmartToolHolder: {qname(data_g, holder)}")


def check_required_runtime_windows(data_g: Graph) -> None:
    windows = set(data_g.subjects(RDF.type, STH.MultiSensorWindow))
    if not windows:
        raise SystemExit(
            "[FAIL] No runtime windows found. Expected subjects like "
            "mis:RuntimeWindow_w10234 with rdf:type sth:MultiSensorWindow."
        )


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def validate_graph(
    data_g: Graph,
    shapes_g: Graph,
    report_txt: Path,
    report_ttl: Path,
    inference: str = "rdfs",
    advanced: bool = True,
    debug: bool = False,
) -> bool:
    conforms, results_graph, results_text = validate(
        data_graph=data_g,
        shacl_graph=shapes_g,
        inference=inference,
        advanced=advanced,
        debug=debug,
        meta_shacl=False,
    )

    report_txt.parent.mkdir(parents=True, exist_ok=True)
    report_ttl.parent.mkdir(parents=True, exist_ok=True)
    report_txt.write_text(results_text, encoding="utf-8")
    results_graph.serialize(destination=report_ttl, format="turtle")

    print("\n===== SHACL Validation Report =====")
    print(results_text)
    print(f"[REPORT] {report_txt}")
    print(f"[REPORT] {report_ttl}")

    return bool(conforms)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate only runtime MultiSensorWindow TTL data with CNC SHACL shapes."
    )
    parser.add_argument(
        "--runtime",
        default="runtime/CNC_runtime_observation.ttl",
        help="Runtime TTL file. Default: runtime/CNC_runtime_observation.ttl",
    )
    parser.add_argument(
        "--shapes",
        default="shapes/CNC_generated_shapes.ttl",
        help="SHACL shapes TTL file. Default: shapes/CNC_generated_shapes.ttl",
    )
    parser.add_argument(
        "--report-txt",
        default="output/runtime_validation_report.txt",
        help="Human-readable validation report output.",
    )
    parser.add_argument(
        "--report-ttl",
        default="output/runtime_validation_report.ttl",
        help="RDF/Turtle validation report output.",
    )
    parser.add_argument(
        "--no-shape-patch",
        action="store_true",
        help="Disable path-alias patching for unit-bearing properties in shapes.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable pySHACL debug output.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    runtime_file = resolve_path(args.runtime)
    shapes_file = resolve_path(args.shapes)
    report_txt = resolve_path(args.report_txt)
    report_ttl = resolve_path(args.report_ttl)

    print(f"[ROOT] {PROJECT_ROOT}")

    raw_runtime_g = parse_required_ttl(runtime_file, "runtime")
    shapes_g = parse_required_ttl(shapes_file, "shapes")

    data_g = keep_only_runtime_layer(raw_runtime_g)
    check_required_runtime_windows(data_g)

    if not args.no_shape_patch:
        changed = patch_shape_paths(shapes_g)
        print(f"[PATCH] Total sh:path replacements: {changed}")

    summarize_runtime(data_g)

    ok = validate_graph(
        data_g=data_g,
        shapes_g=shapes_g,
        report_txt=report_txt,
        report_ttl=report_ttl,
        debug=args.debug,
    )

    if ok:
        print("[PASS] Runtime layer validation passed.")
        raise SystemExit(0)

    print("[FAIL] Runtime layer validation failed.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
