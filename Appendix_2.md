## Appendix 2. Analysis of Continuous Instruction Guiding with Ontology Data
## Evaluation Results

The Llama4 answers were evaluated using the correct-answer column as the reference.

| Evaluation index | Llama4 |
| --- | ---: |
| Answer correctness | 98.2% |
| Completeness | 96.4% |
| Faithfulness | 100.0% |
| Semantic similarity | 97.9% |
| Content-Element F1 | 98.2% |

Completeness measures the proportion of required reference-answer elements included in each response, while faithfulness measures the proportion of generated claims supported by the reference. Answer correctness is calculated as the harmonic mean of completeness and faithfulness. Semantic similarity is evaluated according to meaning equivalence. Content-Element F1 balances content precision and content recall, thereby evaluating factual correctness and coverage without requiring identical wording.

Llama4 produced substantively correct and faithful answers throughout this continuous instruction-guiding sequence. The small reduction in completeness and Content-Element F1 resulted from Question 3, in which MQTT was identified but Wi-Fi was omitted.

Because the retrieved ontology passages are not included, faithfulness represents reference-supported faithfulness rather than strict retrieval-grounded faithfulness.

### Q&A Questions
| NO | Question | Ontology<br>Llama4 | Correct answer |
| --- | --- | --- | --- |
| 1 | Where is the CNC-TMV720 machine located? | TT | TT |
| 2 | Which controller does it use? | Syntec_21 | Syntec_21 |
| 3 | Which Smart Tool Holder models does it use, and what communication method do they use? | BBT40-v2,<br>BT40-v2;<br>MQTT | BBT40-v2<br>BT40-v2<br>MQTT/ Wi-Fi |
| 4 | How do you wake up the Smart Tool Holder before connecting it? | Shake the Smart Tool Holder beyond the G-threshold to wake it. | Shake the Smart Tool Holder so that it exceeds the G-value threshold to wake it up. |
| 5 | How do you start recording Smart Tool Holder sensor data? | START RECORD | Press START RECORD. |
| 6 | When monitoring with the BBT40-v2 Smart Tool Holder, what does it mean if the real-time monitoring status shows Hold? | The current signal is unstable or the connection to the Smart Tool Holder has been interrupted, resulting in incomplete received data. | It means the signal is currently unstable or the connection to the STH has been interrupted, resulting in incomplete received data. |
| 7 | What may be affected if the Smart Tool Holder signal is insufficient during machining? | Connection interrupted or incomplete data. | Connection interruptions or incomplete data. |


