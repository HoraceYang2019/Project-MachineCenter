from __future__ import annotations

"""
validate_knowledge_layer.py

Purpose
-------
Validate only CNC batch knowledge TTL files.

This script intentionally ignores runtime MultiSensorWindow data and static/demo
ontology individuals. It validates only real knowledge nodes whose subject URI
starts with:

    mis:Knowledge_

Expected valid knowledge node example:

    mis:Knowledge_B_ToolWorn
        rdf:type mis:BatchKnowledge ;
        mis:forBatch mis:TMV720_Batch_B ;
        mis:forState mis:ToolWorn ;
        mis:hasKnowledgeCategory "tool_condition" ;
        mis:minTorqueRatio "1.18"^^xsd:double ;
        mis:maxTorqueRatio "1.52"^^xsd:double ;
        mis:confidence "0.94"^^xsd:double .

Run from project root:

    python validation/validate_knowledge_layer.py
"""

from pathlib import Path
from typing import Iterable

from pyshacl import validate
from rdflib import Graph, Literal, Namespace, RDF
from rdflib.namespace import XSD

BASE_DIR = Path(__file__).resolve().parents[1]

ONTOLOGY_DIR = BASE_DIR / "ontology"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
SHAPES_DIR = BASE_DIR / "shapes"
OUTPUT_DIR = BASE_DIR / "output"

# Adjust names if your ontology files use different names.
ONTOLOGY_FILES = [
    ONTOLOGY_DIR / "CNC_ontology_1.ttl",
    ONTOLOGY_DIR / "CNC_ontology_auto_from_schema.ttl",
    ONTOLOGY_DIR / "CNC_ontology.ttl",
]

KNOWLEDGE_FILES = sorted(KNOWLEDGE_DIR.glob("*.ttl"))

SHAPES_FILE = SHAPES_DIR / "TMV720_hybrid_shapes.ttl"

REPORT_TXT = OUTPUT_DIR / "knowledge_validation_report.txt"
REPORT_TTL = OUTPUT_DIR / "knowledge_validation_report.ttl"

STH = Namespace("http://nkust.edu.tw/mislab/cnc/ontology/sth#")
MIS = Namespace("http://nkust.edu.tw/mislab#")


def parse_existing(files: Iterable[Path]) -> Graph:
    g = Graph()
    g.bind("sth", STH)
    g.bind("mis", MIS)
    g.bind("xsd", XSD)

    for f in files:
        if f.exists():
            print(f"[LOAD] {f}")
            g.parse(f, format="turtle")

    return g


def normalize_knowledge_category_literals(g: Graph) -> None:
    """
    Normalize:
        "tool_condition"^^xsd:string
    into:
        "tool_condition"

    This avoids false sh:in mismatch problems in some SHACL engines.
    """
    for s, _, o in list(g.triples((None, MIS.hasKnowledgeCategory, None))):
        if isinstance(o, Literal):
            plain = Literal(str(o))
            if o != plain:
                g.remove((s, MIS.hasKnowledgeCategory, o))
                g.add((s, MIS.hasKnowledgeCategory, plain))


def keep_only_batch_knowledge(g: Graph) -> Graph:
    """
    Keep only actual knowledge nodes.

    This prevents batch registry nodes such as mis:TMV720_Batch_A from being
    validated as mis:BatchKnowledge by mistake.
    """
    out = Graph()
    out.bind("sth", STH)
    out.bind("mis", MIS)
    out.bind("xsd", XSD)

    for s in g.subjects(RDF.type, MIS.BatchKnowledge):
        if not str(s).startswith(str(MIS) + "Knowledge_"):
            continue

        for p, o in g.predicate_objects(s):
            out.add((s, p, o))

    normalize_knowledge_category_literals(out)
    return out


def validate_graph(data_g: Graph, shapes_g: Graph) -> bool:
    conforms, results_graph, results_text = validate(
        data_graph=data_g,
        shacl_graph=shapes_g,
        inference="rdfs",
        advanced=True,
        debug=False,
        meta_shacl=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TXT.write_text(results_text, encoding="utf-8")
    results_graph.serialize(destination=REPORT_TTL, format="turtle")

    print(results_text)
    print(f"[REPORT] {REPORT_TXT}")
    print(f"[REPORT] {REPORT_TTL}")

    return bool(conforms)


def main() -> None:
    raw_g = parse_existing([*ONTOLOGY_FILES, *KNOWLEDGE_FILES])
    data_g = keep_only_batch_knowledge(raw_g)

    shapes_g = parse_existing([SHAPES_FILE])

    print("\n===== Knowledge Layer Summary =====")
    print(f"Knowledge files:       {len(KNOWLEDGE_FILES)}")
    print(f"Filtered triples:      {len(data_g)}")
    print(f"BatchKnowledge nodes:  {len(set(data_g.subjects(RDF.type, MIS.BatchKnowledge)))}")

    if len(data_g) == 0:
        print("[FAIL] No knowledge nodes found. Expected subjects like mis:Knowledge_B_ToolWorn.")
        raise SystemExit(1)

    ok = validate_graph(data_g, shapes_g)

    if ok:
        print("[PASS] Knowledge layer validation passed.")
    else:
        print("[FAIL] Knowledge layer validation failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
