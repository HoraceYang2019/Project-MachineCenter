from __future__ import annotations

"""
Generate CNC/TMV720 SHACL shapes from the uploaded/project TTL files.

This script is intended for the actual CNC/STH case where SHACL shapes should be
generated not only from the core ontology, but also from runtime/knowledge TTLs
that reveal the actual properties used in the pipeline.

Typical project-root usage:

    python scripts/generate_cnc_generated_shapes.py ^
        --ontology ontology/CNC_ontology_1.ttl ^
        --extra-ontology ontology/_ontology_auto_from_schema.ttl ^
        --knowledge knowledge/STH_threshold_knowledge.ttl ^
        --runtime runtime/_runtime_observation.ttl ^
        --selector runtime/_runtime_batch_selector.ttl ^
        --out shapes/CNC_generated_shapes.ttl

If you only want ontology-derived shapes:

    python scripts/generate_cnc_generated_shapes.py ^
        --ontology ontology/CNC_ontology_1.ttl ^
        --out shapes/CNC_generated_shapes.ttl

Dependencies:
    pip install rdflib
"""

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef
from rdflib.namespace import XSD


MIS = Namespace("http://nkust.edu.tw/mislab#")
STH = Namespace("http://nkust.edu.tw/mislab/cnc/ontology/sth#")
SH = Namespace("http://www.w3.org/ns/shacl#")
SHAPE = Namespace("http://nkust.edu.tw/mislab/cnc/shapes#")


# ---------------------------------------------------------------------
# Required property profile for the actual TMV720/STH runtime pipeline.
# These are stronger than ontology-only domain/range extraction.
# ---------------------------------------------------------------------
REQUIRED_PROFILE: dict[URIRef, set[URIRef]] = {
    STH.MultiSensorWindow: {
        MIS.hasWindowId,
        MIS.belongsToProject,
        MIS.belongsToBatch,
        MIS.hasTimestamp,
        STH.hasStartTime,
        STH.hasEndTime,
        STH.alignedWithNCBlock,
        STH.alignedWithToolPathSegment,
        MIS.hasSpindleSpeed,
        MIS.hasFeedRate,
        MIS.hasTorqueRatio,
        MIS.hasVibrationRMS,
        MIS.hasChatterScore,
        MIS.hasMeasuredRa,
        MIS.hasBatteryLevel,
        MIS.hasCommScore,
        MIS.hasMaxTorque,
        MIS.hasSequenceIndex,
        MIS.hasProgressRatio,
        MIS.hasBlockProgressRatio,
    },
    MIS.RuntimeBatchSelector: {
        MIS.activeBatch,
    },
    MIS.STHThresholdKnowledge: {
        MIS.appliesToDevice,
        MIS.hasKnowledgeCategory,
        MIS.observedProperty,
        MIS.inferredProperty,
        MIS.minValue,
        MIS.minInclusive,
        MIS.maxValue,
        MIS.maxExclusive,
        MIS.forState,
        MIS.priority,
        MIS.profileId,
    },
    MIS.STHCompositeRuleKnowledge: {
        MIS.appliesToDevice,
        MIS.hasKnowledgeCategory,
        MIS.inferredProperty,
        MIS.ruleExpression,
        MIS.forState,
        MIS.priority,
        MIS.profileId,
    },
    MIS.BatchKnowledge: {
        MIS.hasKnowledgeCategory,
        MIS.forState,
    },
}

# Properties that normally should not have sh:maxCount 1.
MULTI_VALUE_PROPERTIES: set[URIRef] = {
    RDF.type,
    MIS.activeBatch,
}

# Skip these predicates as SHACL property constraints.
SKIP_PREDICATES: set[URIRef] = {
    RDF.type,
    RDFS.label,
    RDFS.comment,
    OWL.sameAs,
}


@dataclass
class PropertySpec:
    path: URIRef
    datatype: URIRef | None = None
    class_range: URIRef | None = None
    node_kind_iri: bool = False
    node_kind_literal: bool = False
    required: bool = False
    max_count: int | None = None
    sources: set[str] = field(default_factory=set)


def resolve_existing_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    return p if p.exists() else p


def local_name(uri: URIRef) -> str:
    text = str(uri)
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    return text.rstrip("/").rsplit("/", 1)[-1]


def safe_local_name(uri: URIRef) -> str:
    name = local_name(uri)
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "Shape"


def bind_common(g: Graph) -> None:
    g.bind("mis", MIS)
    g.bind("sth", STH)
    g.bind("sh", SH)
    g.bind("shape", SHAPE)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)


def parse_graph(paths: Iterable[Path], label: str) -> Graph:
    g = Graph()
    bind_common(g)
    for path in paths:
        if path is None:
            continue
        if not path.exists():
            raise FileNotFoundError(f"{label} TTL not found: {path}")
        g.parse(path, format="turtle")
    return g


def collect_domain_range_specs(schema_graph: Graph) -> dict[URIRef, dict[URIRef, PropertySpec]]:
    """
    Collect property specs from owl:ObjectProperty/owl:DatatypeProperty plus rdfs:domain/rdfs:range.
    """
    by_class: dict[URIRef, dict[URIRef, PropertySpec]] = defaultdict(dict)

    for prop in set(schema_graph.subjects(RDF.type, OWL.DatatypeProperty)):
        domains = list(schema_graph.objects(prop, RDFS.domain))
        ranges = list(schema_graph.objects(prop, RDFS.range))
        datatype = ranges[0] if ranges else None

        for domain in domains:
            spec = by_class[domain].setdefault(prop, PropertySpec(path=prop))
            if isinstance(datatype, URIRef):
                spec.datatype = datatype
                spec.node_kind_literal = True
            spec.sources.add("ontology-domain-range")

    for prop in set(schema_graph.subjects(RDF.type, OWL.ObjectProperty)):
        domains = list(schema_graph.objects(prop, RDFS.domain))
        ranges = list(schema_graph.objects(prop, RDFS.range))
        class_range = ranges[0] if ranges else None

        for domain in domains:
            spec = by_class[domain].setdefault(prop, PropertySpec(path=prop))
            if isinstance(class_range, URIRef):
                # If range is xsd:* accidentally declared on object property, ignore as sh:class.
                if not str(class_range).startswith(str(XSD)):
                    spec.class_range = class_range
            spec.node_kind_iri = True
            spec.sources.add("ontology-domain-range")

    return by_class


def infer_literal_datatype(literals: list[Literal]) -> URIRef | None:
    datatypes = [lit.datatype for lit in literals if lit.datatype is not None]
    if datatypes:
        # Prefer an explicitly typed datatype. If multiple appear, use the first stable one.
        return datatypes[0]

    # Plain literals default to string.
    if literals:
        return XSD.string

    return None


def collect_observed_specs(data_graph: Graph, schema_graph: Graph) -> dict[URIRef, dict[URIRef, PropertySpec]]:
    """
    Collect property specs from actual individuals in runtime/knowledge TTLs.

    This is important because some properties in the uploaded TTLs are used by data
    but do not have complete rdfs:domain/rdfs:range declarations in the ontology.
    """
    by_class: dict[URIRef, dict[URIRef, PropertySpec]] = defaultdict(dict)

    for subj in set(data_graph.subjects()):
        classes = [c for c in data_graph.objects(subj, RDF.type) if isinstance(c, URIRef)]
        # Skip class declarations themselves.
        if OWL.Class in classes or RDFS.Class in classes:
            continue

        # If no explicit class, there is no sh:targetClass to attach to.
        if not classes:
            continue

        pred_to_objects: dict[URIRef, list] = defaultdict(list)
        for pred, obj in data_graph.predicate_objects(subj):
            if pred in SKIP_PREDICATES:
                continue
            if not isinstance(pred, URIRef):
                continue
            pred_to_objects[pred].append(obj)

        for cls in classes:
            if cls in {OWL.NamedIndividual, OWL.Class, RDFS.Class}:
                continue

            for pred, objects in pred_to_objects.items():
                spec = by_class[cls].setdefault(pred, PropertySpec(path=pred))
                spec.sources.add("observed-data")

                literal_values = [o for o in objects if isinstance(o, Literal)]
                iri_values = [o for o in objects if isinstance(o, URIRef)]

                if literal_values and not spec.datatype:
                    spec.datatype = infer_literal_datatype(literal_values)
                    spec.node_kind_literal = True

                if iri_values:
                    spec.node_kind_iri = True
                    # Try to infer a class range from object rdf:type.
                    object_types = []
                    for o in iri_values:
                        object_types.extend([t for t in data_graph.objects(o, RDF.type) if isinstance(t, URIRef)])
                        object_types.extend([t for t in schema_graph.objects(o, RDF.type) if isinstance(t, URIRef)])

                    object_types = [
                        t for t in object_types
                        if t not in {OWL.NamedIndividual, OWL.Class, RDFS.Class}
                    ]

                    if object_types and not spec.class_range:
                        spec.class_range = object_types[0]

    return by_class


def merge_specs(*spec_maps: dict[URIRef, dict[URIRef, PropertySpec]]) -> dict[URIRef, dict[URIRef, PropertySpec]]:
    merged: dict[URIRef, dict[URIRef, PropertySpec]] = defaultdict(dict)

    for spec_map in spec_maps:
        for cls, props in spec_map.items():
            for prop, spec in props.items():
                out = merged[cls].setdefault(prop, PropertySpec(path=prop))
                out.datatype = out.datatype or spec.datatype
                out.class_range = out.class_range or spec.class_range
                out.node_kind_iri = out.node_kind_iri or spec.node_kind_iri
                out.node_kind_literal = out.node_kind_literal or spec.node_kind_literal
                out.required = out.required or spec.required
                out.max_count = out.max_count if out.max_count is not None else spec.max_count
                out.sources.update(spec.sources)

    # Apply manual required profile.
    for cls, required_props in REQUIRED_PROFILE.items():
        for prop in required_props:
            spec = merged[cls].setdefault(prop, PropertySpec(path=prop))
            spec.required = True
            spec.sources.add("manual-required-profile")
            if prop not in MULTI_VALUE_PROPERTIES:
                spec.max_count = 1

    return merged


def add_property_shape(shapes: Graph, node_shape: URIRef, spec: PropertySpec) -> None:
    ps = BNode()
    shapes.add((node_shape, SH.property, ps))
    shapes.add((ps, SH.path, spec.path))

    if spec.datatype is not None:
        shapes.add((ps, SH.datatype, spec.datatype))
    elif spec.class_range is not None:
        shapes.add((ps, SH["class"], spec.class_range))
    elif spec.node_kind_iri:
        shapes.add((ps, SH.nodeKind, SH.IRI))
    elif spec.node_kind_literal:
        shapes.add((ps, SH.nodeKind, SH.Literal))

    if spec.required:
        shapes.add((ps, SH.minCount, Literal(1, datatype=XSD.integer)))

    if spec.max_count is not None:
        shapes.add((ps, SH.maxCount, Literal(spec.max_count, datatype=XSD.integer)))

    if spec.sources:
        shapes.add((ps, RDFS.comment, Literal("Generated from: " + ", ".join(sorted(spec.sources)))))


def generate_shapes(
    ontology_paths: list[Path],
    data_paths: list[Path],
    out_path: Path,
) -> Graph:
    schema_graph = parse_graph(ontology_paths, "ontology")
    data_graph = parse_graph(data_paths, "data") if data_paths else Graph()
    bind_common(data_graph)

    # Combined graph is used for observed range inference.
    combined = Graph()
    bind_common(combined)
    for t in schema_graph:
        combined.add(t)
    for t in data_graph:
        combined.add(t)

    domain_range_specs = collect_domain_range_specs(schema_graph)
    observed_specs = collect_observed_specs(combined, schema_graph)
    specs = merge_specs(domain_range_specs, observed_specs)

    shapes = Graph()
    bind_common(shapes)

    for cls in sorted(specs.keys(), key=lambda u: str(u)):
        props = {
            p: s for p, s in specs[cls].items()
            if p not in SKIP_PREDICATES
        }

        if not props:
            continue

        node_shape = SHAPE[f"{safe_local_name(cls)}Shape"]
        shapes.add((node_shape, RDF.type, SH.NodeShape))
        shapes.add((node_shape, SH.targetClass, cls))
        shapes.add((node_shape, RDFS.comment, Literal(f"Auto-generated SHACL shape for {cls}")))

        for prop in sorted(props.keys(), key=lambda u: str(u)):
            add_property_shape(shapes, node_shape, props[prop])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shapes.serialize(destination=out_path, format="turtle")

    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CNC_generated_shapes.ttl from ontology, runtime, selector, and STH knowledge TTL files."
    )

    parser.add_argument(
        "--ontology",
        nargs="+",
        default=["ontology/CNC_ontology_1.ttl"],
        help="Core ontology TTL file(s).",
    )
    parser.add_argument(
        "--extra-ontology",
        nargs="*",
        default=[],
        help="Optional ontology/schema-derived TTL file(s), e.g. _ontology_auto_from_schema.ttl.",
    )
    parser.add_argument(
        "--knowledge",
        nargs="*",
        default=[],
        help="Optional knowledge TTL file(s), e.g. STH_threshold_knowledge.ttl.",
    )
    parser.add_argument(
        "--runtime",
        nargs="*",
        default=[],
        help="Optional runtime observation TTL file(s), e.g. _runtime_observation.ttl.",
    )
    parser.add_argument(
        "--selector",
        nargs="*",
        default=[],
        help="Optional runtime selector TTL file(s), e.g. _runtime_batch_selector.ttl.",
    )
    parser.add_argument(
        "--out",
        default="shapes/CNC_generated_shapes.ttl",
        help="Output SHACL shapes TTL path.",
    )

    args = parser.parse_args()

    ontology_paths = [Path(p) for p in (args.ontology + args.extra_ontology)]
    data_paths = [Path(p) for p in (args.knowledge + args.runtime + args.selector)]

    shapes = generate_shapes(
        ontology_paths=ontology_paths,
        data_paths=data_paths,
        out_path=Path(args.out),
    )

    print("[OK] Generated CNC SHACL shapes")
    print(f"     Output: {args.out}")
    print(f"     Ontology TTLs: {len(ontology_paths)}")
    for p in ontology_paths:
        print(f"       - {p}")
    print(f"     Data/knowledge TTLs: {len(data_paths)}")
    for p in data_paths:
        print(f"       - {p}")
    print(f"     Shapes triples: {len(shapes)}")


if __name__ == "__main__":
    main()
