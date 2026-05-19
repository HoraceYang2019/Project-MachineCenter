from __future__ import annotations

"""
rdf_native_infer_sparql.py

Purpose
-------
Load CNC ontology, knowledge, runtime TTL files, apply RDF-native SPARQL
CONSTRUCT rules, and print inferred runtime results.

This version is aligned with the updated rule set:

    rules/01_infer_tool_condition.rq
    rules/02_infer_surface_quality.rq
    rules/03_infer_process_state.rq
    rules/04_infer_sth_state.rq

It prints four inference families:

    1. Tool condition
    2. Process state
    3. Surface quality
    4. Smart Tool Holder states

Run from the project root or from the folder containing this script:

    python rdf_native_infer_sparql.py

Optional examples:

    python rdf_native_infer_sparql.py --rules rules --out output/CNC_Runtime_Inferred_Aligned.ttl

    python rdf_native_infer_sparql.py --strict-files
"""

import argparse
from pathlib import Path
from typing import Iterable

from rdflib import Graph, Namespace, RDF, URIRef

# ---------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------
STH = Namespace("http://nkust.edu.tw/mislab/cnc/ontology/sth#")
MIS = Namespace("http://nkust.edu.tw/mislab#")


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------
def infer_base_dir() -> Path:
    """
    Use the script folder as the project base.
    This matches your current project structure where the script is located
    directly under MC_sematic_adv/.
    """
    return Path(__file__).resolve().parent


def resolve(base: Path, path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else base / p


def existing_files(files: Iterable[Path], strict: bool = False) -> list[Path]:
    found: list[Path] = []

    for f in files:
        if f.exists():
            found.append(f)
        else:
            msg = f"[SKIP] Missing file: {f}"
            if strict:
                raise FileNotFoundError(msg)
            print(msg)

    return found


# ---------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------
def parse_ttl_files(g: Graph, files: Iterable[Path], label: str) -> None:
    for f in files:
        print(f"[LOAD:{label}] {f}")
        g.parse(f, format="turtle")


def load_graph(
    ontology_files: list[Path],
    knowledge_files: list[Path],
    runtime_files: list[Path],
    strict_files: bool = False,
) -> Graph:
    g = Graph()

    g.bind("sth", STH)
    g.bind("mis", MIS)

    ontology_existing = existing_files(ontology_files, strict=strict_files)
    knowledge_existing = existing_files(knowledge_files, strict=strict_files)
    runtime_existing = existing_files(runtime_files, strict=strict_files)

    parse_ttl_files(g, ontology_existing, "ontology")
    parse_ttl_files(g, knowledge_existing, "knowledge")
    parse_ttl_files(g, runtime_existing, "runtime")

    return g


# ---------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------
def apply_construct_rule(g: Graph, rule_file: Path) -> int:
    """
    Apply one SPARQL CONSTRUCT rule and add newly constructed triples.

    Returns:
        Number of newly added triples.
    """
    query_text = rule_file.read_text(encoding="utf-8")
    constructed = g.query(query_text)

    added = 0
    for triple in constructed:
        if len(triple) != 3:
            continue
        if triple not in g:
            g.add(triple)
            added += 1

    return added


def apply_rules(g: Graph, rule_dir: Path, rule_files: list[str] | None = None) -> int:
    if rule_files:
        rules = [rule_dir / name for name in rule_files]
    else:
        rules = sorted(rule_dir.glob("*.rq"))

    if not rules:
        raise FileNotFoundError(f"No .rq files found in rule directory: {rule_dir}")

    total_added = 0

    print("\n===== Applying SPARQL CONSTRUCT Rules =====")
    for rule in rules:
        if not rule.exists():
            print(f"[SKIP] Missing rule: {rule}")
            continue

        try:
            added = apply_construct_rule(g, rule)
            total_added += added
            print(f"{rule.name}: added {added} triples")
        except Exception as exc:
            print(f"[ERROR] Failed to apply {rule.name}")
            print(f"        {type(exc).__name__}: {exc}")
            raise

    return total_added


# ---------------------------------------------------------------------
# Debugging helpers
# ---------------------------------------------------------------------
def count_subjects(g: Graph, rdf_type: URIRef) -> int:
    return len(set(g.subjects(RDF.type, rdf_type)))


def print_debug_summary(g: Graph) -> None:
    print("\n===== Loaded Graph Debug Summary =====")
    print(f"Graph triples:                      {len(g)}")
    print(f"Runtime MultiSensorWindow nodes:    {count_runtime_windows(g)}")
    print(f"mis:BatchKnowledge rules:           {count_subjects(g, MIS.BatchKnowledge)}")
    print(f"mis:STHThresholdKnowledge rules:    {count_subjects(g, MIS.STHThresholdKnowledge)}")
    print(f"mis:STHCompositeKnowledge rules:    {count_subjects(g, MIS.STHCompositeKnowledge)}")
    print(f"mis:STHCompositeRuleKnowledge rules:{count_subjects(g, MIS.STHCompositeRuleKnowledge)}")
    print(f"Runtime has torque ratio triples:   {len(list(g.triples((None, MIS.hasTorqueRatio, None))))}")
    print(f"Runtime has measured Ra triples:    {len(list(g.triples((None, MIS.hasMeasuredRa, None))))}")
    print(f"Runtime has chatter observed triples:{len(list(g.triples((None, MIS.hasChatterScoreObserved, None))))}")
    print(f"Runtime has chatter score triples:  {len(list(g.triples((None, MIS.hasChatterScore, None))))}")
    print(f"Runtime has battery triples:        {len(list(g.triples((None, MIS.hasBatteryLevel, None))))}")
    print(f"Runtime has comm-score triples:     {len(list(g.triples((None, MIS.hasCommScore, None))))}")
    print(f"Runtime has max-torque triples:     {len(list(g.triples((None, MIS.hasMaxTorque, None))))}")


def count_runtime_windows(g: Graph) -> int:
    return len(list(runtime_windows(g)))


def runtime_windows(g: Graph) -> list[URIRef]:
    windows: list[URIRef] = []

    for window in g.subjects(RDF.type, STH.MultiSensorWindow):
        if str(window).startswith(str(MIS) + "RuntimeWindow_"):
            windows.append(window)

    return sorted(set(windows), key=lambda x: str(x))


def qname(g: Graph, value) -> str:
    if value is None:
        return "None"

    try:
        return g.namespace_manager.normalizeUri(value)
    except Exception:
        return str(value)


def first_value(g: Graph, subject: URIRef, predicates: Iterable[URIRef]):
    for p in predicates:
        value = g.value(subject, p)
        if value is not None:
            return value
    return None


# ---------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------
def print_results(g: Graph) -> None:
    print("\n===== Inferred Results =====\n")

    windows = runtime_windows(g)

    if not windows:
        print("No runtime MultiSensorWindow instances found.")
        return

    for window in windows:
        holder = first_value(
            g,
            window,
            [
                MIS.hasSmartToolHolder,
                STH.usesSmartToolHolder,
            ],
        )

        # Four desired inference outputs.
        tool = g.value(window, MIS.hasToolCondition)
        process = g.value(window, MIS.hasProcessState)
        surface = g.value(window, MIS.hasSurfaceQuality)

        # STH component states and final state.
        battery = g.value(window, MIS.hasBatteryState)
        comm = g.value(window, MIS.hasCommunicationState)
        load = g.value(window, MIS.hasLoadState)
        sth_state = g.value(window, MIS.hasSTHState)

        print(f"Window: {qname(g, window)}")
        print(f"  SmartToolHolder:      {qname(g, holder)}")
        print(f"   - Battery state:       {qname(g, battery)}")
        print(f"   - Communication state: {qname(g, comm)}")
        print(f"   - Load state:          {qname(g, load)}")
        print(f"   - STH state:           {qname(g, sth_state)}")
        print("")
        print(f"  Tool condition:       {qname(g, tool)}")
        print(f"  Process state:        {qname(g, process)}")
        print(f"  Surface quality:      {qname(g, surface)}")
        print("")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_default_paths(base: Path) -> tuple[list[Path], list[Path], list[Path]]:
    ontology_files = [
        base / "ontology" / "CNC_ontology_1.ttl",

        # Support both names because your project has used both variants.
        base / "ontology" / "CNC_ontology_auto_from_schema.ttl",
        base / "ontology" / "_ontology_auto_from_schema.ttl",
    ]

    knowledge_files = [
        base / "knowledge" / "batch_registry.ttl",
        base / "knowledge" / "batch_A.ttl",
        base / "knowledge" / "batch_B.ttl",
        base / "knowledge" / "batch_C.ttl",
        base / "knowledge" / "STH_threshold_knowledge.ttl",
    ]

    runtime_files = [
        base / "runtime" / "_runtime_batch_selector.ttl",

        # Main runtime file used by your current validation report.
        base / "runtime" / "CNC_runtime_observation.ttl",

        # Fallback name used in earlier generation.
        base / "runtime" / "_runtime_observation.ttl",
    ]

    return ontology_files, knowledge_files, runtime_files


def main() -> None:
    base = infer_base_dir()

    default_ontology, default_knowledge, default_runtime = build_default_paths(base)

    parser = argparse.ArgumentParser(
        description="Apply CNC RDF-native SPARQL CONSTRUCT inference rules."
    )

    parser.add_argument(
        "--base",
        default=str(base),
        help="Project base directory. Default: folder containing this script.",
    )

    parser.add_argument(
        "--rules",
        default="rules",
        help="Rule directory relative to base. Default: rules",
    )

    parser.add_argument(
        "--out",
        default="output/CNC_Runtime_Inferred_Aligned.ttl",
        help="Output inferred TTL path relative to base.",
    )

    parser.add_argument(
        "--strict-files",
        action="store_true",
        help="Fail if an expected ontology/knowledge/runtime file is missing.",
    )

    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Do not print loaded graph debug summary.",
    )

    parser.add_argument(
        "--rule-file",
        action="append",
        default=None,
        help=(
            "Specific rule filename to run, relative to --rules. "
            "Can be repeated. If omitted, all *.rq files are run in sorted order."
        ),
    )

    args = parser.parse_args()

    base = Path(args.base).resolve()
    rule_dir = resolve(base, args.rules)
    output_ttl = resolve(base, args.out)

    ontology_files, knowledge_files, runtime_files = build_default_paths(base)

    print(f"[BASE] {base}")

    g = load_graph(
        ontology_files=ontology_files,
        knowledge_files=knowledge_files,
        runtime_files=runtime_files,
        strict_files=args.strict_files,
    )

    if not args.no_debug:
        print_debug_summary(g)

    print(f"\nLoaded graph size before inference: {len(g)}")

    total_added = apply_rules(g, rule_dir, rule_files=args.rule_file)

    output_ttl.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=output_ttl, format="turtle")

    print("\n===== Inference Summary =====")
    print(f"Total added triples: {total_added}")
    print(f"Saved: {output_ttl}")
    print(f"Final graph size: {len(g)}")

    print_results(g)


if __name__ == "__main__":
    main()
