## Appendix 1. Comparison examples of Different Language Models for Ontology Question Answering

| NO | Question | Ontology<br>Breeze2 | Ontology<br>Llama4 | Correct answer |
| --- | --- | --- | --- | --- |
| 23 | When analyzing the simulation results of FlexSplineNCFile_01, which charts can be viewed on the Simulation Analysis page? | Coordinate X,<br>Coordinate Y,<br>Coordinate Z | Moment Z,<br>Moment Y,<br>Coordinate X, Coordinate Y, Coordinate Z | Moment Z、Moment Y、Coordinate X、Coordinate Y、Coordinate Z |
| 24 | How can the time range of the monitoring data for FlexSplineNCFile_01 be determined? | To determine the time range of the monitoring data for FlexSplineNCFile_01, you can refer to the partA_op10.nc file. The time range information is available in the OPCUA_NCFile and CNC-TMV720 fields. | The time range of the monitoring data for FlexSplineNCFile_01 can be determined by checking the Start Time and End Time in the history view. | Review the Start Time and End Time in the history data to determine the exact time range of the test. |
| 28 | When 2026FlexSpline is used with the BBT40-v2 Smart Tool Holder, which operating mode may the holder enter if it has not been used for a long time? | Smart Tool Holder = BBT40-v2;NC File = partA_op10.nc | Sleep mode. | Sleep mode. |

