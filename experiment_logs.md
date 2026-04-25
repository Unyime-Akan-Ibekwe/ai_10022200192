# Manual Experiment Logs

**Student:** Unyime Akan Ibekwe  
**Index:** 10022200192  
**Course:** CS4241 — Introduction to Artificial Intelligence  
**Date:** April 2026  



## Log 1 — Chunking Strategy Experiment

**Objective:**  
Find the best chunk size for retrieval quality  

**Setup:**  
Same query tested with different chunk sizes in `chunk_text()`  

**Query:**  
"Who won 2020 election?"

| Chunk Size | Overlap | Result Quality |
|-----------|--------|---------------|
| 200 chars | 50     | Too small — answers split across chunks, incomplete |
| 500 chars | 100    | Good — full sentences retrieved, coherent answer |
| 1000 chars| 200    | Too large — irrelevant content mixed into chunks |

**Conclusion:**  
500 characters with 100 overlap gave the best balance between context and precision. This was chosen as the final chunking strategy.



## Log 2 — Retrieval Quality Experiment

**Objective:**  
Compare vector-only retrieval vs hybrid retrieval  

**Query:**  
"What are the fiscal risks in the 2025 budget?"

### Vector-Only Retrieval (Before Hybrid)
- Returned general budget text with no specific risk mentions  
- **Score:** 0.61  
- **Answer Quality:** Vague  

### Hybrid Retrieval (After Keyword + Domain Boost)
- Returned chunks mentioning:
  - Debt  
  - Arrears  
  - Deficit  
  - Energy sector risks  
- **Score:** 0.89  
- **Answer Quality:** Specific and factual  

**Conclusion:**  
Hybrid retrieval significantly improved relevance for domain-specific queries. Pure vector search missed keyword-heavy factual chunks.



## Log 3 — Query Expansion Experiment

**Objective:**  
Test whether query expansion improves retrieval for short queries  

**Query:**  
"budget risks"

### Without Expansion
- Retrieved 3 chunks, only 1 relevant  
- Missing:
  - Energy sector risks  
  - Cocoa revenue  
  - Financial sector risks  

### With Expansion
Expanded query:  
`debt arrears deficit risks challenges energy cocoa financial sector`

- Retrieved 5 chunks, all relevant  
- Included:
  - Energy sector risks  
  - Cocoa revenue shortfalls  
  - Deficit figures  

**Conclusion:**  
Query expansion significantly improved recall for short domain queries. Essential for budget-related questions.



## Log 4 — Prompt Engineering Experiment

**Objective:**  
Compare different prompt designs on the same query  

**Query:**  
"Who won the 2020 Ghana presidential election?"

### Prompt Version 1 (Strict)
> Answer ONLY using the context. If not found say "Answer not found."

- **Result:** "Answer not found in context"  
- **Issue:** Too restrictive — ignored partial matches  

### Prompt Version 2 (Flexible)
> Use the context to answer. If partially relevant, use what you can and say so.

- **Result:** Named John Dramani Mahama with regional wins  
- **Outcome:** Much more useful  

**Conclusion:**  
Strict prompts lead to false negatives. Flexible prompts with honesty instructions provide better answers while controlling hallucination.



## Log 5 — Context Window Experiment

**Objective:**  
Test how context size affects answer quality  

**Query:**  
"Summarise the key highlights of the 2025 Ghana budget"

| Max Context Chars | Chunks Passed | Answer Quality |
|------------------|--------------|---------------|
| 1000             | 2            | Incomplete — only GDP mentioned |
| 2500             | 4            | Better — includes revenue & expenditure |
| 6000             | 7            | Best — includes fiscal policy, oil, cocoa, debt |

**Conclusion:**  
A larger context window (6000 characters) produced significantly richer answers for broad queries. This was selected as the final configuration.



## Log 6 — Adversarial Query Experiment

**Objective:**  
Test system robustness on ambiguous and misleading queries  

### Query 1: "Who won?" (Ambiguous)

- **RAG Response:** Listed regional winners and asked for clarification  
- **Hallucination:** None  
- **Verdict:** Handled well 



### Query 2:  
"How much money did Ghana make from oil in the 2025 budget?" (Misleading)

- **RAG Response:**  
  - Retrieved petroleum production data  
  - Mentioned:
    - 48.24 million barrels  
    - Benchmark price: $74.70/barrel  
  - Clearly stated full revenue figure not available  

- **Hallucination:** None  
- **Verdict:** Handled well 

---

## Log 7 — RAG vs Pure LLM Comparison

**Objective:**  
Provide evidence-based comparison between RAG and non-retrieval LLM  

---

### Query 1: "Who won the 2020 Ghana election?"

| Metric | RAG System | Pure LLM |
|-------|-----------|---------|
| Answer | Named candidates + regional data | May name winner, lacks vote data |
| Grounded in Data | Yes (CSV) | No |
| Hallucination Risk | Low | High |
| Verifiable | Yes | No |

---

### Query 2: "What are the risks in the 2025 Ghana budget?"

| Metric | RAG System | Pure LLM |
|-------|-----------|---------|
| Answer | Specific risks (debt, energy, cocoa) | Generic economic risks |
| Grounded in Data | Yes (PDF) | No |
| Hallucination Risk | Low | High |
| Verifiable | Yes | No |

---

**Final Conclusion:**  
RAG systems provide more accurate, grounded, and verifiable responses compared to pure LLMs. Retrieval significantly reduces hallucination and improves domain specificity.

---
