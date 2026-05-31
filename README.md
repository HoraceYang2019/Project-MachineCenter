1. **Prepare ontology**:
   
   location: ontology/CNC_ontology_*.ttl

---
2. **Prepare editable component \ batch thresholds**:
   
   location: knowledge/thresholds_component.csv \ batch_*.csv

3. **Generate component \batch knowledge TTL**:
   
   location: knowledge/component_threshold_knowledge.ttl \ batch_*.ttl
  > python scripts/generate_batch_ttl_from_threshold_csv.py <br>
  > python scripts/generate_sth_ttl_from_threshold_csv.py
  
  *Validate inferred TTL:* location: shapes\CNC_gerated_shapes.ttl
  > python scripts/generate_cnc_generated_shapes.py

---  
4. **Prepare runtime observation json**:
   
   location: sample/_runtime_w10233.json

   
5.  **Generate runtime observation TTL**:
  
  location: runtime/CNC_runtime_observation.ttl 
  > python runtime/runtime_observation_ttl_from_windows_json_multi.py <br>
  
   *Validate inferred TTL: shapes\CNC_generated_shapes.ttl*

---
6. **Prepare SPARQL inference**:<br>
  location: rules/01_infer_tool_condition.rq <br>
  location: rules/02_infer_surface_quality.rq <br>
  location: rules/03_infer_process_state.rq <br>
  location: rules/04_infer_sth_state.rq <br>
  *Validate inferred TTL: shapes\CNC_generated_shapes.ttl*


7. **Run inference**:
   > python rdf_native_infer_sparql.py
