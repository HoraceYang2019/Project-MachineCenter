import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef
from rdflib.namespace import XSD


DEFAULT_NAMESPACES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "sth": "http://nkust.edu.tw/mislab/cnc/ontology/sth#",
    "mis": "http://nkust.edu.tw/mislab#"
}


def local_name(uri: URIRef) -> str:
    text = str(uri)
    if "#" in text:
        return text.split("#")[-1]
    return text.rstrip("/").split("/")[-1]


def clean_property_name(name: str) -> str:
    """
    Convert hasTorqueRatio -> torque_ratio
    Convert hasSpindleSpeed(rpm/min) -> spindle_speed
    Convert hasMeasuredRa(um) -> measured_ra
    """
    name = re.sub(r"\(.*?\)", "", name)

    if name.startswith("has"):
        name = name[3:]
    elif name.startswith("is"):
        name = name[2:]

    # Convert CamelCase to snake_case
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)

    return name.strip("_").lower()


def infer_json_path(prop: URIRef, domain: Optional[URIRef]) -> str:
    """
    Heuristic JSON path inference.

    This is only a draft. You should manually verify the result.
    """
    prop_name = clean_property_name(local_name(prop))
    domain_name = local_name(domain) if domain else ""

    process_keywords = {
        "spindle_speed",
        "feed_rate",
        "cutting_speed",
        "depth_of_cut",
        "radial_depth",
        "axial_depth"
    }

    feature_keywords = {
        "torque_ratio",
        "vibration_rms",
        "acoustic_rms",
        "chatter_score",
        "measured_ra",
        "force_rms",
        "temperature",
        "roughness"
    }

    time_keywords = {
        "timestamp",
        "start_time",
        "end_time"
    }

    alignment_keywords = {
        "nc_block",
        "tool_path_segment"
    }

    if prop_name in process_keywords:
        return f"process.{prop_name}"

    if prop_name in feature_keywords:
        return f"features.{prop_name}"

    if prop_name in time_keywords:
        if prop_name == "start_time":
            return "time_window.start"
        if prop_name == "end_time":
            return "time_window.end"
        return "timestamp"

    if prop_name in alignment_keywords:
        return f"alignment.{prop_name}"

    if prop_name in {"project", "batch"}:
        return prop_name

    if prop_name == "window_id":
        return "window_id"

    if "MultiSensorWindow" in domain_name:
        return prop_name

    return prop_name


def qname(g: Graph, uri: URIRef) -> str:
    try:
        return g.namespace_manager.qname(uri)
    except Exception:
        return str(uri)


def datatype_to_qname(g: Graph, datatype: Optional[URIRef]) -> Optional[str]:
    if datatype is None:
        return None
    return qname(g, datatype)


def collect_namespaces(g: Graph) -> Dict[str, str]:
    namespaces = dict(DEFAULT_NAMESPACES)

    for prefix, ns in g.namespaces():
        if prefix:
            namespaces[str(prefix)] = str(ns)

    return namespaces


def first_or_none(values: List[URIRef]) -> Optional[URIRef]:
    return values[0] if values else None


def generate_binding(
    ontology_file: Path,
    target_class_qname: str,
    out_file: Path
) -> None:
    g = Graph()
    g.parse(ontology_file, format="turtle")

    namespaces = collect_namespaces(g)

    field_mappings = []

    # Datatype properties
    for prop in sorted(g.subjects(RDF.type, OWL.DatatypeProperty), key=lambda x: str(x)):
        domains = list(g.objects(prop, RDFS.domain))
        ranges = list(g.objects(prop, RDFS.range))

        domain = first_or_none(domains)
        datatype = first_or_none(ranges)

        # Keep only properties whose domain is the target class if possible.
        # If domain is missing, include it but mark required false.
        domain_qname = qname(g, domain) if domain else None

        if domain_qname is not None and domain_qname != target_class_qname:
            continue

        field_mappings.append({
            "json_path": infer_json_path(prop, domain),
            "rdf_property": qname(g, prop),
            "rdf_datatype": datatype_to_qname(g, datatype) or "xsd:string",
            "kind": "datatype",
            "required": False,
            "max_count": 1,
            "source": "auto-generated-from-ontology"
        })

    # Object properties
    for prop in sorted(g.subjects(RDF.type, OWL.ObjectProperty), key=lambda x: str(x)):
        domains = list(g.objects(prop, RDFS.domain))
        ranges = list(g.objects(prop, RDFS.range))

        domain = first_or_none(domains)
        obj_class = first_or_none(ranges)

        domain_qname = qname(g, domain) if domain else None

        if domain_qname is not None and domain_qname != target_class_qname:
            continue

        prop_name = clean_property_name(local_name(prop))
        obj_class_qname = qname(g, obj_class) if obj_class else None

        field_mappings.append({
            "json_path": infer_json_path(prop, domain),
            "rdf_property": qname(g, prop),
            "rdf_object_template": "mis:{value}",
            "rdf_object_class": obj_class_qname,
            "kind": "object",
            "required": False,
            "max_count": 1,
            "source": "auto-generated-from-ontology"
        })

    binding = {
        "binding_name": "CNC MultiSensorWindow Runtime Binding Draft",
        "version": "0.1.0",
        "description": (
            "Draft binding automatically generated from ontology TTL. "
            "Please manually verify json_path, required, object templates, and units."
        ),
        "namespaces": namespaces,
        "target_class": {
            "json_path": "$",
            "rdf_class": target_class_qname
        },
        "subject_template": {
            "base_prefix": "mis",
            "template": "RuntimeWindow_{window_id}"
        },
        "required_fields": [],
        "field_mappings": field_mappings
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(binding, f, indent=2, ensure_ascii=False)

    print(f"[OK] Draft binding generated: {out_file}")
    print(f"[INFO] Field mappings: {len(field_mappings)}")
    print("[WARNING] Please manually verify json_path and required fields.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a draft CNC runtime binding JSON from ontology TTL."
    )

    parser.add_argument(
        "--ontology",
        default="ontology/CNC_ontology_1.ttl",
        help="Input ontology TTL file."
    )

    parser.add_argument(
        "--target-class",
        default="sth:MultiSensorWindow",
        help="Target RDF class for runtime window binding."
    )

    parser.add_argument(
        "--out",
        default="mappings/CNC_runtime_binding.draft.json",
        help="Output binding JSON file."
    )

    args = parser.parse_args()

    generate_binding(
        ontology_file=Path(args.ontology),
        target_class_qname=args.target_class,
        out_file=Path(args.out)
    )


if __name__ == "__main__":
    main()