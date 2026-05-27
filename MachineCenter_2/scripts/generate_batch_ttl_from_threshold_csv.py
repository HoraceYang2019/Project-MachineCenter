from __future__ import annotations

"""
scripts/generate_batch_ttl_from_threshold_csv.py

Run from the PROJECT ROOT:

    python scripts/generate_batch_ttl_from_threshold_csv.py

This default command reads all:
    knowledge/thresholds_TMV720_batch_*.csv

and generates:
    knowledge/TMV720_knowledge_batch_*.ttl

Single-file mode:

    python scripts/generate_batch_ttl_from_threshold_csv.py \
        --csv knowledge/thresholds_batch_B.csv \
        --out knowledge/knowledge_batch_B.ttl
"""

import argparse
import csv
from pathlib import Path
from typing import Dict

from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import XSD

PROJECT_ROOT = Path.cwd()

MIS = Namespace("http://nkust.edu.tw/mislab#")
STH = Namespace("http://nkust.edu.tw/mislab/cnc/ontology/sth#")

PREFIX_MAP = {"mis": MIS, "sth": STH}


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def qname_to_uri(qname: str) -> URIRef:
    qname = str(qname).strip()
    if ":" not in qname:
        raise ValueError(f"Expected prefixed name such as mis:ToolWorn or sth:Diagnosis_xxx, got: {qname}")
    prefix, local = qname.split(":", 1)
    if prefix not in PREFIX_MAP:
        raise ValueError(f"Unsupported prefix '{prefix}' in {qname}. Supported prefixes: {sorted(PREFIX_MAP)}")
    return PREFIX_MAP[prefix][local]


def require(row: Dict[str, str], key: str) -> str:
    value = row.get(key, "")
    if value is None:
        value = ""
    value = str(value).strip()
    if value == "":
        raise ValueError(f"Missing required CSV field '{key}' in row: {row}")
    return value


def literal_double(value: str, field_name: str) -> Literal:
    try:
        return Literal(float(value), datatype=XSD.double)
    except ValueError as exc:
        raise ValueError(f"CSV field '{field_name}' must be numeric, got: {value}") from exc


def add_knowledge_row(g: Graph, row: Dict[str, str]) -> None:
    batch_id = require(row, "batch_id")
    knowledge_id = require(row, "knowledge_id")
    category = require(row, "category")
    state_uri = require(row, "state_uri")
    min_property = require(row, "min_property")
    min_value = require(row, "min_value")
    max_property = require(row, "max_property")
    max_value = require(row, "max_value")
    confidence = require(row, "confidence")
    dataset_id = require(row, "dataset_id")
    description = str(row.get("description", "") or "").strip()

    if not knowledge_id.startswith("Knowledge_"):
        raise ValueError(f"knowledge_id must start with 'Knowledge_' for safe SHACL targeting. Got: {knowledge_id}")

    subject = MIS[knowledge_id]
    g.remove((subject, None, None))

    g.add((subject, RDF.type, MIS.BatchKnowledge))
    g.add((subject, MIS.forBatch, MIS[batch_id]))
    g.add((subject, MIS.forState, qname_to_uri(state_uri)))
    g.add((subject, MIS.hasKnowledgeCategory, Literal(category)))  # plain literal for sh:in compatibility
    g.add((subject, qname_to_uri(min_property), literal_double(min_value, "min_value")))
    g.add((subject, qname_to_uri(max_property), literal_double(max_value, "max_value")))
    g.add((subject, MIS.confidence, literal_double(confidence, "confidence")))
    g.add((subject, MIS.derivedFromDataset, Literal(dataset_id)))
    if description:
        g.add((subject, MIS.hasDescription, Literal(description)))


def generate_one(csv_file: Path, output_ttl: Path) -> Graph:
    csv_file = resolve_path(csv_file)
    output_ttl = resolve_path(output_ttl)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    required_columns = {
        "batch_id", "knowledge_id", "category", "state_uri",
        "min_property", "min_value", "max_property", "max_value",
        "confidence", "dataset_id",
    }

    g = Graph()
    g.bind("mis", MIS)
    g.bind("sth", STH)
    g.bind("xsd", XSD)

    with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_file} is missing required CSV columns: {sorted(missing)}")
        row_count = 0
        for row in reader:
            add_knowledge_row(g, row)
            row_count += 1

    output_ttl.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=output_ttl, format="turtle")

    print("[OK] Generated knowledge TTL")
    print(f"     CSV:     {csv_file}")
    print(f"     TTL:     {output_ttl}")
    print(f"     Rows:    {row_count}")
    print(f"     Triples: {len(g)}")
    return g


def default_output_for_csv(csv_file: Path) -> Path:
    name = csv_file.name.replace("thresholds_", "").replace(".csv", ".ttl")
    return PROJECT_ROOT / "knowledge" / name


def discover_csv_files() -> list[Path]:
    return sorted((PROJECT_ROOT / "knowledge").glob("thresholds_batch_*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CNC BatchKnowledge TTL from CSV.")
    parser.add_argument("--csv", help="Input CSV. If omitted, all knowledge/thresholds_batch_*.csv files are processed.")
    parser.add_argument("--out", help="Output TTL. Optional when --csv is supplied.")
    args = parser.parse_args()

    if args.csv:
        csv_file = resolve_path(args.csv)
        output_ttl = resolve_path(args.out) if args.out else default_output_for_csv(csv_file)
        generate_one(csv_file, output_ttl)
        return

    csv_files = discover_csv_files()
    if not csv_files:
        raise SystemExit("[FAIL] No CSV files found: knowledge/thresholds_batch_*.csv")
    for csv_file in csv_files:
        generate_one(csv_file, default_output_for_csv(csv_file))


if __name__ == "__main__":
    main()
