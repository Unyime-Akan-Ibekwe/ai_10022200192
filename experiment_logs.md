Manual Experiment Logs
Student: Unyime Akan Ibekwe
Index: 10022200192
Course: CS4241 — Introduction to Artificial Intelligence
Date: April 2026

Log 1 — Chunking Strategy Experiment
Objective: Find the best chunk size for retrieval quality
Setup: Same query tested with different chunk sizes in chunk_text()
Chunk SizeOverlapQueryResult Quality200 chars50"Who won 2020 election?"Too small — answers split across chunks, incomplete500 chars100"Who won 2020 election?"Good — full sentences retrieved, coherent answer1000 chars200"Who won 2020 election?"Too large — irrelevant content mixed into chunks
Conclusion: 500 chars with 100 overlap gave the best balance between context and precision. Chosen as final chunking strategy.

Log 2 — Retrieval Quality Experiment
Objective: Compare vector-only retrieval vs hybrid retrieval
Query: "What are the fiscal risks in the 2025 budget?"
Vector-only result (before hybrid):

Returned general budget text with no specific risk mentions
Score: 0.61
Answer quality: Vague

Hybrid retrieval result (after adding keyword + domain boost):

Returned chunks explicitly mentioning debt, arrears, deficit, energy sector risks
Score: 0.89
Answer quality: Specific and factual

Conclusion: Hybrid retrieval significantly improved relevance for domain-specific queries. Pure vector search missed keyword-heavy factual chunks.

Log 3 — Query Expansion Experiment
Objective: Test whether query expansion improves retrieval for short queries
Query: "budget risks"
Without expansion:

Retrieved 3 chunks, only 1 relevant
Missing: energy sector, cocoa, financial sector risks

With expansion (added "debt arrears deficit risks challenges energy cocoa financial sector"):

Retrieved 5 chunks, all relevant
Included: energy sector risks, cocoa revenue shortfalls, deficit figures

Conclusion: Query expansion directly improved recall for short domain queries. Essential for budget-related questions where users use simple terms.

Log 4 — Prompt Engineering Experiment
Objective: Compare different prompt designs on same query
Query: "Who won the 2020 Ghana presidential election?"
Prompt Version 1 (strict):
Answer ONLY using the context. If not found say "Answer not found."
Result: "Answer not found in context" — too restrictive, ignored partial matches
Prompt Version 2 (flexible):
Use the context to answer. If partially relevant, use what you can and say so.
Result: Named John Dramani Mahama with regional wins listed — much more useful
Conclusion: Strict prompts cause false negatives. Flexible prompts with honesty instructions produce better answers while still controlling hallucination.

Log 5 — Context Window Experiment
Objective: Test how context size affects answer quality
Query: "Summarise the key highlights of the 2025 Ghana budget"
Max Context CharsChunks PassedAnswer Quality10002 chunksIncomplete — only mentioned GDP25004 chunksBetter — added revenue and expenditure60007 chunksBest — covered fiscal policy, oil, cocoa, debt
Conclusion: Larger context window (6000 chars) produced significantly richer answers for broad summary questions. Set as final value.

Log 6 — Adversarial Query Experiment
Objective: Test system robustness on ambiguous and misleading queries
Query 1: "Who won?" (ambiguous — no year or region)

RAG Response: Listed regional winners across years, asked for clarification
Hallucination: None — only stated what was in context
Verdict: Handled well ✅

Query 2: "How much money did Ghana make from oil in the 2025 budget?" (misleading — implies direct figure exists)

RAG Response: Retrieved petroleum production data (48.24M barrels, US$74.70/barrel benchmark) but correctly stated full revenue figure not available
Hallucination: None — was transparent about missing data
Verdict: Handled well ✅


Log 7 — RAG vs Pure LLM Comparison
Objective: Evidence-based comparison of RAG vs no retrieval
Query: "Who won the 2020 Ghana election?"
MetricRAG SystemPure LLM (no retrieval)AnswerNamed candidates with regional dataMay name correct winner but no vote dataGrounded in data?Yes — from CSVNo — from training memoryHallucination riskLowHighVerifiable?Yes — chunks shownNo
Query: "What are the risks in the 2025 Ghana budget?"
MetricRAG SystemPure LLMAnswerSpecific risks from PDF — debt, energy, cocoaGeneric economic risks, not Ghana-specificGrounded in data?Yes — from budget PDFNoHallucination riskLowHighVerifiable?YesNo
Overall Conclusion: RAG system consistently outperformed pure LLM on domain-specific factual queries. The retrieval layer grounded all answers in actual documents, eliminating hallucination on specific figures.
