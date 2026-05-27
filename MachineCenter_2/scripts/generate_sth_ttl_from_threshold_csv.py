from __future__ import annotations

"""
generate_sth_knowledge_ttl_from_csv.py

Generate STH threshold/composite knowledge TTL from thresholds_STH.csv.

Expected CSV columns
--------------------
profile_id,knowledge_id,rule_type,category,observed_property,
min_value,min_inclusive,max_value,max_exclusive,
inferred_property,state_uri,state_class,confidence,priority,
condition_expression,description

Rule types
----------
threshold:
    Numeric threshold rule.
    Example:
        observed_property = mis:hasBatteryLevel
        inferred_property = mis:hasBatteryState
        state_uri = mis:Normal_Battery
        min_value = 1.58
        max_value = 9.99

composite:
    Logical rule assembled from threshold condition IDs.
    Example:
        condition_expression =
        (STH_Normal_Battery) and (STH_Comm_Stable) and
        (STH_Normal_Load or STH_Light_Load)

Run from project root
---------------------
python scripts/generate_sth_knowledge_ttl_from_csv.py ^
  --csv knowledge/thresholds_STH.csv ^
  --out knowledge/STH_threshold_knowledge.ttl ^
  --ontology ontology/CNC_ontology_1.ttl

Optional device-specific generation
-----------------------------------
If your STH inference rule requires device-specific knowledge, add:

python scripts/generate_sth_knowledge_ttl_from_csv.py ^
  --csv knowledge/thresholds_STH.csv ^
  --out knowledge/STH_threshold_knowledge.ttl ^
  --device-id STH_002
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import OWL, XSD


# ---------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------
MIS = Namespace("http://nkust.edu.tw/mislab#")
STH = Namespace("http://nkust.edu.tw/mislab/cnc/ontology/sth#")

PREFIX_MAP = {
    "mis": MIS,
    "sth": STH,
    "rdf": RDF,
    "rdfs": RDFS,
    "owl": OWL,
    "xsd": XSD,
}


REQUIRED_COLUMNS = {
    "profile_id",
    "knowledge_id",
    "rule_type",
    "category",
    "observed_property",
    "min_value",
    "min_inclusive",
    "max_value",
    "max_exclusive",
    "inferred_property",
    "state_uri",
    "state_class",
    "confidence",
    "priority",
    "condition_expression",
    "description",
}


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def is_empty(value) -> bool:
    return value is None or str(value).strip() == "" or str(value).strip().lower() == "nan"


def clean_id(value: str) -> str:
    """
    Convert CSV identifiers into safe RDF local names.

    Examples:
        STH-2_Normal_Battery -> STH_2_Normal_Battery
        STH_Normal_Battery   -> STH_Normal_Battery
    """
    value = str(value).strip()
    value = value.replace("-", "_")
    value = re.sub(r"[^A-Za-z0-9_]", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def uri_from_prefixed(value: str) -> URIRef:
    """
    Convert a prefixed name or absolute IRI into URIRef.

    Examples:
        mis:hasBatteryLevel -> http://nkust.edu.tw/mislab#hasBatteryLevel
        sth:STH_002         -> http://nkust.edu.tw/mislab/cnc/ontology/sth#STH_002
    """
    value = str(value).strip()

    if value.startswith("http://") or value.startswith("https://"):
        return URIRef(value)

    if ":" not in value:
        return MIS[clean_id(value)]

    prefix, local = value.split(":", 1)

    if prefix not in PREFIX_MAP:
        raise ValueError(f"Unknown prefix '{prefix}' in value '{value}'.")

    return PREFIX_MAP[prefix][local]


def qname(g: Graph, uri: URIRef) -> str:
    try:
        return g.namespace_manager.qname(uri)
    except Exception:
        return str(uri)


def profile_uri(profile_id: str) -> URIRef:
    return MIS[clean_id(profile_id)]


def knowledge_uri(knowledge_id: str) -> URIRef:
    return MIS[clean_id(knowledge_id)]


def condition_uri(condition_id: str) -> URIRef:
    return MIS[clean_id(condition_id)]


def device_uri(device_id: str) -> URIRef:
    """
    Normalize device IDs.

    Examples:
        STH-2   -> sth:STH_2
        STH_002 -> sth:STH_002
    """
    return STH[clean_id(device_id)]


def bool_literal(value, default: bool = True) -> Literal:
    if is_empty(value):
        return Literal(default, datatype=XSD.boolean)

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y"}:
        return Literal(True, datatype=XSD.boolean)

    if text in {"false", "0", "no", "n"}:
        return Literal(False, datatype=XSD.boolean)

    return Literal(default, datatype=XSD.boolean)


def double_literal(value) -> Literal | None:
    if is_empty(value):
        return None
    return Literal(float(value), datatype=XSD.double)


def integer_literal(value) -> Literal | None:
    if is_empty(value):
        return None
    return Literal(int(float(value)), datatype=XSD.integer)


def string_literal(value) -> Literal:
    return Literal(str(value).strip(), datatype=XSD.string)


def extract_condition_ids(expression: str) -> list[str]:
    """
    Extract condition IDs from a logical expression.

    Example:
        (STH_Normal_Battery) and (STH_Comm_Stable)
        -> ["STH_Normal_Battery", "STH_Comm_Stable"]
    """
    if is_empty(expression):
        return []

    # IDs are normally inside parentheses.
    ids = re.findall(r"\(([A-Za-z0-9_\-]+)\)", str(expression))

    # Fallback for expressions without parentheses.
    if not ids:
        reserved = {"and", "or", "not", "AND", "OR", "NOT"}
        ids = [
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-]*", str(expression))
            if token not in reserved
        ]

    # Preserve order while removing duplicates.
    seen = set()
    out = []
    for item in ids:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


# ---------------------------------------------------------------------
# Optional ontology checking
# ---------------------------------------------------------------------
def load_ontology_terms(ontology_file: Path | None) -> set[URIRef]:
    if ontology_file is None:
        return set()

    if not ontology_file.exists():
        raise FileNotFoundError(f"Ontology file not found: {ontology_file}")

    g = Graph()
    g.parse(ontology_file, format="turtle")

    terms: set[URIRef] = set()

    for s, p, o in g:
        if isinstance(s, URIRef):
            terms.add(s)
        if isinstance(p, URIRef):
            terms.add(p)
        if isinstance(o, URIRef):
            terms.add(o)

    return terms


def collect_csv_rdf_terms(rows: list[dict[str, str]]) -> set[URIRef]:
    terms: set[URIRef] = set()

    rdf_columns = [
        "observed_property",
        "inferred_property",
        "state_uri",
        "state_class",
    ]

    for row in rows:
        for col in rdf_columns:
            value = row.get(col, "")
            if not is_empty(value):
                terms.add(uri_from_prefixed(value))

    return terms


def warn_missing_terms(rows: list[dict[str, str]], ontology_terms: set[URIRef], g: Graph) -> None:
    if not ontology_terms:
        return

    csv_terms = collect_csv_rdf_terms(rows)
    missing = sorted([term for term in csv_terms if term not in ontology_terms], key=str)

    if missing:
        print("[WARNING] Some CSV RDF terms were not found in the ontology:")
        for term in missing:
            print(f"  - {qname(g, term)}")
    else:
        print("[OK] All CSV RDF terms were found in the ontology.")


# ---------------------------------------------------------------------
# RDF generation
# ---------------------------------------------------------------------
def bind_namespaces(g: Graph) -> None:
    g.bind("mis", MIS)
    g.bind("sth", STH)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)


def add_profile(g: Graph, profile_id: str) -> URIRef:
    p = profile_uri(profile_id)
    g.add((p, RDF.type, MIS.STHKnowledgeProfile))
    g.add((p, RDFS.label, string_literal(profile_id)))
    return p


def add_condition_node(
    g: Graph,
    condition_id: str,
    state: URIRef | None = None,
    state_class: URIRef | None = None,
) -> URIRef:
    c = condition_uri(condition_id)
    g.add((c, RDF.type, MIS.STHCondition))
    g.add((c, RDFS.label, string_literal(condition_id)))

    if state is not None:
        g.add((c, MIS.forState, state))

    if state_class is not None:
        g.add((c, MIS.hasStateClass, state_class))

    return c


def add_common_fields(
    g: Graph,
    row: dict[str, str],
    k: URIRef,
    device_id: str | None = None,
) -> None:
    profile_id = row["profile_id"].strip()
    knowledge_id = row["knowledge_id"].strip()
    rule_type = row["rule_type"].strip().lower()
    category = row["category"].strip()
    inferred_property = row.get("inferred_property", "").strip()
    state_uri_value = row.get("state_uri", "").strip()
    state_class_value = row.get("state_class", "").strip()
    confidence_value = row.get("confidence", "").strip()
    priority_value = row.get("priority", "").strip()
    description = row.get("description", "").strip()

    p = add_profile(g, profile_id)

    g.add((k, RDF.type, MIS.STHKnowledgeRule))
    g.add((k, MIS.hasProfile, p))
    g.add((k, MIS.hasProfileId, string_literal(profile_id)))
    g.add((k, MIS.hasKnowledgeId, string_literal(knowledge_id)))
    g.add((k, MIS.hasRuleType, string_literal(rule_type)))

    if category:
        g.add((k, MIS.hasKnowledgeCategory, string_literal(category)))

    if device_id:
        g.add((k, MIS.appliesToDevice, device_uri(device_id)))

    if inferred_property:
        g.add((k, MIS.inferredProperty, uri_from_prefixed(inferred_property)))

    if state_uri_value:
        state = uri_from_prefixed(state_uri_value)
        g.add((k, MIS.forState, state))

        if state_class_value:
            state_class = uri_from_prefixed(state_class_value)
            g.add((state, RDF.type, state_class))

    conf = double_literal(confidence_value)
    if conf is not None:
        g.add((k, MIS.confidence, conf))

    pri = integer_literal(priority_value)
    if pri is not None:
        g.add((k, MIS.priority, pri))

    if description:
        g.add((k, RDFS.comment, string_literal(description)))


def add_threshold_rule(
    g: Graph,
    row: dict[str, str],
    k: URIRef,
) -> None:
    g.add((k, RDF.type, MIS.STHThresholdKnowledge))

    observed_property = row.get("observed_property", "").strip()
    if is_empty(observed_property):
        raise ValueError(f"Threshold rule '{row['knowledge_id']}' has empty observed_property.")

    g.add((k, MIS.observedProperty, uri_from_prefixed(observed_property)))

    min_lit = double_literal(row.get("min_value", ""))
    max_lit = double_literal(row.get("max_value", ""))

    if min_lit is not None:
        g.add((k, MIS.minValue, min_lit))
        g.add((k, MIS.minInclusive, bool_literal(row.get("min_inclusive", ""), default=True)))

    if max_lit is not None:
        g.add((k, MIS.maxValue, max_lit))
        g.add((k, MIS.maxExclusive, bool_literal(row.get("max_exclusive", ""), default=True)))

    state = uri_from_prefixed(row["state_uri"]) if not is_empty(row.get("state_uri", "")) else None
    state_class = uri_from_prefixed(row["state_class"]) if not is_empty(row.get("state_class", "")) else None

    condition = add_condition_node(
        g=g,
        condition_id=row["knowledge_id"],
        state=state,
        state_class=state_class,
    )

    g.add((k, MIS.hasCondition, condition))


def add_composite_rule(
    g: Graph,
    row: dict[str, str],
    k: URIRef,
) -> None:
    # Support both class names used in your earlier rule versions.
    g.add((k, RDF.type, MIS.STHCompositeKnowledge))
    g.add((k, RDF.type, MIS.STHCompositeRuleKnowledge))

    expr = row.get("condition_expression", "").strip()
    if is_empty(expr):
        raise ValueError(f"Composite rule '{row['knowledge_id']}' has empty condition_expression.")

    g.add((k, MIS.conditionExpression, string_literal(expr)))

    # Alias used by some older scripts/rules.
    g.add((k, MIS.ruleExpression, string_literal(expr)))

    state = uri_from_prefixed(row["state_uri"]) if not is_empty(row.get("state_uri", "")) else None
    state_class = uri_from_prefixed(row["state_class"]) if not is_empty(row.get("state_class", "")) else None

    composite_condition = add_condition_node(
        g=g,
        condition_id=row["knowledge_id"],
        state=state,
        state_class=state_class,
    )
    g.add((k, MIS.hasCondition, composite_condition))

    for cid in extract_condition_ids(expr):
        c = add_condition_node(g, cid)
        g.add((k, MIS.requiresCondition, c))


def read_csv_rows(csv_file: Path) -> list[dict[str, str]]:
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        rows = []
        for i, row in enumerate(reader, start=2):
            if all(is_empty(v) for v in row.values()):
                continue
            row = {k: "" if v is None else str(v).strip() for k, v in row.items()}
            if is_empty(row.get("knowledge_id", "")):
                raise ValueError(f"Row {i} has empty knowledge_id.")
            if is_empty(row.get("rule_type", "")):
                raise ValueError(f"Row {i} has empty rule_type.")
            rows.append(row)

    return rows


def validate_internal_references(rows: list[dict[str, str]]) -> None:
    known_ids = {row["knowledge_id"].strip() for row in rows}
    errors: list[str] = []

    for row in rows:
        if row["rule_type"].strip().lower() != "composite":
            continue

        refs = extract_condition_ids(row.get("condition_expression", ""))
        for ref in refs:
            if ref not in known_ids:
                errors.append(
                    f"Composite rule '{row['knowledge_id']}' references unknown condition '{ref}'."
                )

    if errors:
        raise ValueError("Invalid composite references:\n  " + "\n  ".join(errors))

    print("[OK] Composite condition references are synchronized with CSV knowledge_id values.")


def generate_ttl(
    csv_file: Path,
    out_file: Path,
    ontology_file: Path | None = None,
    device_id: str | None = None,
) -> Graph:
    rows = read_csv_rows(csv_file)

    g = Graph()
    bind_namespaces(g)

    ontology_terms = load_ontology_terms(ontology_file)
    warn_missing_terms(rows, ontology_terms, g)
    validate_internal_references(rows)

    threshold_count = 0
    composite_count = 0

    for row in rows:
        rule_type = row["rule_type"].strip().lower()
        k = knowledge_uri(row["knowledge_id"])

        add_common_fields(g, row, k, device_id=device_id)

        if rule_type == "threshold":
            add_threshold_rule(g, row, k)
            threshold_count += 1
        elif rule_type == "composite":
            add_composite_rule(g, row, k)
            composite_count += 1
        else:
            raise ValueError(
                f"Unsupported rule_type '{rule_type}' for knowledge_id '{row['knowledge_id']}'. "
                "Expected 'threshold' or 'composite'."
            )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=out_file, format="turtle")

    print("[OK] Generated STH knowledge TTL")
    print(f"     Input CSV       : {csv_file}")
    print(f"     Output TTL      : {out_file}")
    if ontology_file:
        print(f"     Ontology checked: {ontology_file}")
    if device_id:
        print(f"     Applies to      : {device_id}")
    else:
        print("     Applies to      : profile-level knowledge, no specific device")
    print(f"     Total rows      : {len(rows)}")
    print(f"     Threshold rules : {threshold_count}")
    print(f"     Composite rules : {composite_count}")
    print(f"     Triples         : {len(g)}")

    return g


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate STH threshold/composite knowledge TTL from thresholds_STH.csv."
    )

    parser.add_argument(
        "--csv",
        default="knowledge/thresholds_STH.csv",
        help="Input STH threshold CSV file.",
    )

    parser.add_argument(
        "--out",
        default="knowledge/STH_threshold_knowledge.ttl",
        help="Output STH knowledge TTL file.",
    )

    parser.add_argument(
        "--ontology",
        default=None,
        help="Optional ontology TTL used only to warn about unknown CSV RDF terms.",
    )

    parser.add_argument(
        "--device-id",
        default=None,
        help=(
            "Optional device ID, e.g. STH_002 or STH-2. "
            "If supplied, every generated rule gets mis:appliesToDevice."
        ),
    )

    args = parser.parse_args()

    ontology_file = Path(args.ontology) if args.ontology else None

    generate_ttl(
        csv_file=Path(args.csv),
        out_file=Path(args.out),
        ontology_file=ontology_file,
        device_id=args.device_id,
    )


if __name__ == "__main__":
    main()
