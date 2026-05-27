from __future__ import annotations

"""
scripts/ontology_ttl_from_csv.py

Run from the PROJECT ROOT:

    python scripts/ontology_ttl_from_csv.py

Default:
    reads  schema/CNCknowledge_csv_schema.json  if it exists
    writes ontology/_ontology_auto_from_csv.ttl
"""

import argparse
import json
from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL
from rdflib.namespace import XSD

PROJECT_ROOT = Path.cwd()
MIS = Namespace("http://nkust.edu.tw/mislab#")
STH = Namespace("http://nkust.edu.tw/mislab/cnc/ontology/sth#")


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def add_class(g: Graph, cls, label: str) -> None:
    g.add((cls, RDF.type, OWL.Class))
    g.add((cls, RDFS.label, Literal(label)))


def add_object_property(g: Graph, prop, domain, range_, label: str) -> None:
    g.add((prop, RDF.type, OWL.ObjectProperty))
    g.add((prop, RDFS.domain, domain))
    g.add((prop, RDFS.range, range_))
    g.add((prop, RDFS.label, Literal(label)))


def add_datatype_property(g: Graph, prop, domain, range_, label: str) -> None:
    g.add((prop, RDF.type, OWL.DatatypeProperty))
    g.add((prop, RDFS.domain, domain))
    g.add((prop, RDFS.range, range_))
    g.add((prop, RDFS.label, Literal(label)))


def read_schema_columns(schema_file: Path) -> set[str]:
    if not schema_file.exists():
        return set()
    data: dict[str, Any] = json.loads(schema_file.read_text(encoding="utf-8"))
    props = data.get("properties", {})
    return set(props.keys()) if isinstance(props, dict) else set()


def build_ontology(schema_columns: set[str]) -> Graph:
    g = Graph()
    g.bind("mis", MIS)
    g.bind("sth", STH)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    add_class(g, MIS.Batch, "Batch")
    add_class(g, MIS.BatchKnowledge, "Batch Knowledge")
    add_class(g, MIS.ToolCondition, "Tool Condition")
    add_class(g, MIS.SurfaceQualityState, "Surface Quality State")
    add_class(g, STH.ProcessState, "Process State")
    add_class(g, MIS.CSVColumn, "CSV Column")

    add_object_property(g, MIS.forBatch, MIS.BatchKnowledge, MIS.Batch, "for batch")
    add_object_property(g, MIS.forState, MIS.BatchKnowledge, OWL.Thing, "for state")

    add_datatype_property(g, MIS.hasKnowledgeCategory, MIS.BatchKnowledge, XSD.string, "knowledge category")
    add_datatype_property(g, MIS.confidence, MIS.BatchKnowledge, XSD.double, "confidence")
    add_datatype_property(g, MIS.derivedFromDataset, MIS.BatchKnowledge, XSD.string, "derived from dataset")
    add_datatype_property(g, MIS.hasDescription, MIS.BatchKnowledge, XSD.string, "description")

    for prop, label in [
        (MIS.minTorqueRatio, "minimum torque ratio"),
        (MIS.maxTorqueRatio, "maximum torque ratio"),
        (MIS.minMeasuredRa, "minimum measured Ra"),
        (MIS.maxMeasuredRa, "maximum measured Ra"),
        (MIS.minChatterScore, "minimum chatter score"),
        (MIS.maxChatterScore, "maximum chatter score"),
    ]:
        add_datatype_property(g, prop, MIS.BatchKnowledge, XSD.double, label)

    for col in sorted(schema_columns):
        col_node = MIS[f"CSVColumn_{col}"]
        g.add((col_node, RDF.type, MIS.CSVColumn))
        g.add((col_node, RDFS.label, Literal(col)))
        g.add((col_node, MIS.usedByPipeline, Literal("CNC BatchKnowledge Schema")))

    return g


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CNC ontology extension from CSV schema.")
    parser.add_argument("--schema", default="schema/CNC_knowledge_csv_schema.json")
    parser.add_argument("--out", default="ontology/CNC_ontology_auto_from_schema.ttl")
    args = parser.parse_args()

    schema_file = resolve_path(args.schema)
    output_file = resolve_path(args.out)
    g = build_ontology(read_schema_columns(schema_file))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=output_file, format="turtle")
    print("[OK] Generated ontology extension")
    print(f"     Schema: {schema_file if schema_file.exists() else 'not found; defaults used'}")
    print(f"     Output: {output_file}")
    print(f"     Triples: {len(g)}")


if __name__ == "__main__":
    main()
