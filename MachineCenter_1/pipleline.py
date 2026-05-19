import scripts.generate_batch_ttl_from_threshold_csv as knowledge_ttl
import scripts.generate_sth_ttl_from_threshold_csv as sth_ttl
import scripts.ontology_ttl_from_schema as ontology_ttl

import scripts.runtime_observation_ttl_from_windows_json_multi as runtime_ttl
import scripts.generate_cnc_generated_shapes as shapes_cnc

import validation.validate_hybrid_stack as validate_stack
import rdf_native_infer_sparql as infer_runtime

if __name__ == '__main__':
    #--- generates knowledge/batch_*.ttl from CSV files
    knowledge_ttl.main()
    
    #--- generates sth knowledge/sth_knowledge.ttl from CSV file
    sth_ttl.main()
    
    #--- generates ontology/_ontology_auto_from_schema.ttl from CSV schema file
    ontology_ttl.main() 

    #--- generates runtime/CNC_runtime_observation.ttl from windows JSON file
    runtime_ttl.main() 

    #--- generates shapes/CNC_hybrid_shapes.ttl from ontology and runtime TTL files
    shapes_cnc.main()
    
    #--- runs SHACL validation on the knowledge and runtime TTL files
    validate_stack.main()

    #--- runs SPARQL-based inference on the combined ontology + knowledge + runtime graph, 
    # and saves inferred triples to output/inferred_runtime.ttl
    infer_runtime.main() 