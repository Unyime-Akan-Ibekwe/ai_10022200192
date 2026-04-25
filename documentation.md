# Project Documentation - Academic City RAG Assistant
**Student:** Unyime-Akan Ibekwe  
**Index Number:** 10022200192  
**Course:** CS4241 - Introduction to Artificial Intelligence  
**Lecturer:** Godwin N. Danso  
**Year:** 2026  

---

## 1. Project Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot built for Academic City University. It allows users to ask questions about Ghana Election Results and the 2025 Ghana Budget Statement. The system was built entirely from scratch without using end-to-end frameworks like LangChain or LlamaIndex. All core components including chunking, embedding, retrieval, and prompt construction were implemented manually.

The application is deployed at:
https://unyime-akan-ibekwe-ai-10022200192-ui-uupix0.streamlit.app/

---

## 2. System Architecture

The system follows a standard RAG pipeline with custom enhancements:

**User Query → Query Expansion → Hybrid Retrieval → Context Selection → Prompt Construction → Groq LLM → Response**

### Components:
- **rag.py** — Loads and cleans datasets, chunks text, builds FAISS index, handles hybrid retrieval and query expansion
- **ui.py** — Streamlit frontend that orchestrates the full pipeline, displays retrieved chunks, scores, prompt, and final response

---

## 3. Part A: Data Engineering

### Datasets Used:
1. Ghana Election Results CSV — regional election results from 1992 to 2020 covering candidates, parties, votes, and percentages
2. 2025 Ghana Budget Statement PDF — 251 pages of fiscal policy, revenue, expenditure, and economic data

### Data Cleaning:
- Filled missing values with empty strings using `fillna("")`
- Converted the Votes column to numeric after stripping commas, since values like "1,234,567" were read as strings by pandas
- Converted the Votes(%) column to numeric after stripping percentage symbols
- Skipped blank PDF pages during extraction to avoid null errors

```python
df["Votes"] = pd.to_numeric(df["Votes"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
df["Votes(%)"] = pd.to_numeric(df["Votes(%)"].astype(str).str.replace("%", ""), errors="coerce").fillna(0)
```

### Chunking Strategy:

**CSV Chunking:**
- Each row was converted directly to a structured text string rather than chunked by character count
- Format: `"Year {year}, Region {region}, Candidate {candidate}, Party {party}, Votes {votes}, Percentage {pct}"`
- Justification: Each row is a discrete data point. Converting row-by-row preserves the exact structure needed for election queries without risk of splitting related figures across chunks

**PDF Chunking:**
- Chunk size: 500 characters
- Overlap: 100 characters
- Split method: sentence-level (split on ". ")
- Justification: Budget documents have dense policy text. 500 characters captures one complete policy point. 100-character overlap prevents important phrases split across chunk boundaries from being missed during retrieval

### Chunking Impact on Retrieval:
Early testing with 200-character chunks produced fragmented context that led to incomplete answers. At 500 characters with 100 overlap, retrieved chunks contained enough context for the LLM to form coherent answers. Sentence-level splitting was preferred over fixed character splits to avoid cutting mid-sentence.

---

## 4. Part B: Custom Retrieval System

### Embedding Pipeline:
- Model: `all-MiniLM-L6-v2` from sentence-transformers
- Embedding dimension: 384
- All chunks embedded at startup and stored in FAISS
- Embeddings are L2-normalised before indexing, making inner product equivalent to cosine similarity

### Vector Storage:
- FAISS `IndexFlatIP` used for exact inner product search
- Chosen because the dataset (a few thousand chunks) is well within the range where exact search is computationally trivial and more accurate than approximate methods

### Top-K Retrieval:
- Default top-k = 8 for retrieval, extended to 12 in the final pipeline
- Each result returns the chunk text and similarity score

### Hybrid Search (Keyword + Vector):
A hybrid scoring function combines three signals:

| Signal | Weight | Description |
|---|---|---|
| Vector Score | 0.5 | Semantic similarity from FAISS inner product |
| Keyword Score | 0.3 | Proportion of query keywords found in the chunk |
| Domain Boost | 0.5 (capped at 1.0) | Rule-based boost for domain-relevant terminology |

Final score formula: final_score = (0.5 × vector_score) + (0.3 × keyword_score) + (0.5 × domain_boost)

### Query Expansion:
Query expansion adds related terms to improve recall. Detected keywords trigger additional search queries:

| Trigger Keyword | Expansion Added |
|---|---|
| risk | challenges problems threats |
| inflation | price increase cost of living |
| growth | gdp economy expansion |
| budget | fiscal policy government spending |
| fiscal | debt arrears deficit risks challenges energy cocoa financial sector |
| won / winner / election | national winner total votes nationwide result |

### Failure Case and Fix:
**Failure:** Querying "who won?" without specifying a year returned only regional winners because national winner chunks were not being prioritised.

**Root Cause:** National winner chunks had no distinguishing prefix to boost their keyword score above regional chunks.

**Fix:** National winner chunks were prefixed with "NATIONAL ELECTION RESULT:" and a query expansion rule for election keywords was added to ensure these chunks rank higher during retrieval.

---

## 5. Part C: Prompt Engineering

### Prompt Template Design:
The prompt template was designed with three goals: inject retrieved context cleanly, prevent hallucination, and produce clear readable answers.

You are a helpful assistant answering questions about Ghana Elections and the 2025 Ghana Budget.
Use the context below to answer the question.
If the answer is in the context, answer clearly and directly.
If the context is partially relevant, use what you can and say so.
CONTEXT:
{context}
QUESTION:
{query}
ANSWER:

### Hallucination Control:
The prompt instructs the model to answer only from provided context. An earlier strict version used "If not found, say answer not found in context" which caused the model to refuse answering even when partial information was available. The revised prompt allows partial answers, which better serves users with imprecise queries.

### Context Window Management:
The `manage_context_window` function accumulates chunks by character count until the limit is reached. Lower-scoring chunks that would exceed the limit are dropped.

| Version | Max Characters | Outcome |
|---|---|---|
| v1 | 2500 | Too few chunks — LLM lacked sufficient context |
| v2 (current) | 6000 | Sufficient context while staying within model limits |

### Prompt Experiment Results:

| Experiment | Prompt Variation | Result |
|---|---|---|
| v1 | Strict: answer ONLY from context, bullet points required | Returned "Answer not found in context" for many valid queries |
| v2 | Relaxed: use what you can, partial answers allowed | More complete and useful responses |
| v3 (current) | Domain-aware intro + partial answers allowed | Best results — grounded answers in natural language |

---

## 6. Part D: Full RAG Pipeline

### Pipeline Stages:
1. **Query received** from Streamlit text input
2. **Query expansion** adds related terms based on detected keywords
3. **FAISS retrieval** returns top-k chunks with similarity scores
4. **Hybrid scoring** re-ranks results using vector score, keyword score, and domain boost
5. **Context selection** filters chunks to stay within 6000 characters
6. **Prompt construction** injects context and query into the template
7. **Groq LLM** generates response using `llama-3.3-70b-versatile`
8. **Response displayed** in Streamlit with retrieved chunks, scores, and final prompt visible

### Logging at Each Stage:
| Stage | What is Displayed in UI |
|---|---|
| Retrieval | All retrieved chunks with similarity scores |
| Context Selection | Chunks passed forward after window filtering |
| Prompt | Full final prompt shown in a code block |
| Response | Generated answer rendered in a styled dark box |

### LLM Model Selection:
| Model Tried | Outcome |
|---|---|
| google/flan-t5-base (HuggingFace) | Poor quality, token limit too low |
| HuggingFaceH4/zephyr-7b-beta | 404 — model moved/gated on free tier |
| mistralai/Mistral-7B-Instruct-v0.1 | 404 — endpoint deprecated |
| llama-3.3-70b-versatile (Groq) | Working, fast, high quality — current choice |

---

## 7. Part E: Critical Evaluation and Adversarial Testing

### Adversarial Query 1: "Who won?"
- **RAG Response:** Listed regional winners across multiple years (Rawlings 1996, Mills 2000/2004, Kuffour 2000/2004) and asked for clarification on year and region. Did not hallucinate a winner.
- **Pure LLM Response:** Would likely confidently name a recent figure or invent a plausible-sounding result with no grounding in the actual data.

| Metric | RAG System | Pure LLM |
|---|---|---|
| Accurate? | Partially — correct names, no national answer | No — would guess or hallucinate |
| Hallucination? | No — only stated what was in context | Yes — likely invents confident answer |
| Consistent? | Yes — honest about ambiguity | No |

**Fix implemented:** Query expansion appends "national winner total votes nationwide result" when election keywords are detected, improving retrieval of national winner chunks for specific queries.

### Adversarial Query 2: "How much money did Ghana make from oil in the 2025 budget?"
- **RAG Response:** Retrieved petroleum chunks and returned real figures — 48.24 million barrels crude oil production in 2024 and benchmark price of US$74.70 per barrel. Honestly stated it could not compute a direct revenue figure since 2025 production totals were not in the document.
- **Pure LLM Response:** Would likely invent a specific dollar figure with no grounding in the actual budget document.

| Metric | RAG System | Pure LLM |
|---|---|---|
| Accurate? | Partially — real figures retrieved, gaps acknowledged | No — would fabricate a number |
| Hallucination? | No — stated when full answer unavailable | Yes — high risk of invented figures |
| Consistent? | Yes — grounded in PDF context | No |

**Fix implemented:** Prompt instructs model to use available information and state honestly when a complete answer cannot be provided.

### Overall RAG vs Pure LLM Comparison:
| Dimension | RAG System | Pure LLM |
|---|---|---|
| Hallucination Rate | Low — grounded in retrieved context | High — generates without evidence |
| Accuracy | Partial to good depending on query specificity | Unreliable for domain-specific data |
| Transparency | High — shows chunks, scores, prompt | None |
| Best suited for | Factual domain questions | General open-ended questions |

---

## 8. Part F: Architecture and System Design

### Architecture Diagram:
*(See Architecture.png in repository)*

The pipeline runs in two phases:

**Offline Phase (startup):**
Data Sources → Load & Clean → Chunk → Embed → FAISS Index (cached by Streamlit)

**Online Phase (query time):**
User → UI → Query Expansion → Embedding → FAISS → Top-K → Hybrid Scoring → Context Selection → Prompt → LLM → Response

### Component Interaction:
| Component | Interacts With | Method |
|---|---|---|
| User Interface | Query stage | Streamlit text input and button |
| Data Processing Pipeline | Vector Database | Runs at startup, populates FAISS index |
| Embedding | Vector Database | Encodes query, searches index |
| Vector Database | Top-K | Returns scored chunk indices |
| Top-K + Hybrid Scoring | Context | Passes re-ranked chunks for filtering |
| Context | Prompt | Injects filtered text into template |
| Prompt | LLM | Sends constructed string via Groq API |
| LLM | Response | Returns generated text to UI |

### Why This Design is Suitable:
- Pre-building the FAISS index at startup and caching it means every query is fast without recomputation
- FAISS IndexFlatIP is suitable for datasets under 100k vectors — our chunk count is well within this range
- Groq free tier with llama-3.3-70b provides high quality responses without cost or GPU requirements
- Hybrid retrieval is necessary for this domain because election data is highly structured and keyword-specific — pure vector search alone is insufficient
- Streamlit Cloud deployment makes the app accessible from any browser without installation

---

## 9. Part G: Innovation — Domain-Specific Scoring Function

### Feature Description:
A custom domain-specific scoring function was implemented inside the hybrid retrieval pipeline to re-rank chunks based on their relevance to the two knowledge domains — Ghana Elections and the 2025 Ghana Budget.

### How It Works:
The domain boost is computed per chunk by checking for domain-critical keywords:

| Category | Keywords | Boost |
|---|---|---|
| Budget risks | risk, debt, inflation, challenge, threat | +0.3 |
| Budget figures | budget, fiscal, revenue, expenditure, GDP, deficit | +0.3 |
| Natural resources | petroleum, oil, energy, cocoa, gold | +0.2 |
| Election results | winner, national, votes, candidate, party | +0.3 |
| Election entities | NDC, NPP, electoral, region, election | +0.2 |
| Statistical content | Any chunk containing digits or numbers | +0.1 |

Boost is capped at 1.0. Final score: final_score = (0.5 × vector_score) + (0.3 × keyword_score) + (0.5 × domain_boost

### Why This is Novel:
Standard RAG uses only vector similarity. Adding a domain scoring layer makes the system more accurate for narrow-domain applications where certain topics are always more relevant than others. The numerical boost specifically rewards chunks containing statistics, which are more likely to hold direct factual answers.

### Evidence of Improvement:
For the query "How much money did Ghana make from oil?", without domain boost the system retrieved generic budget text. With domain boost, petroleum and revenue chunks scored significantly higher, surfacing the crude oil production figure (48.24 million barrels) and benchmark price (US$74.70/barrel) directly.

### Limitation and Future Work:
The keyword list is manually curated and may miss synonyms. A future improvement would be to learn domain weights from user feedback, evolving the scoring function from static rules into an adaptive retrieval ranker.

---

## 10. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.14 |
| UI | Streamlit |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS (IndexFlatIP) |
| LLM | Groq API (llama-3.3-70b-versatile) |
| PDF Parsing | pdfplumber |
| Data Processing | pandas, numpy |
| Deployment | Streamlit Cloud |
| Version Control | GitHub |

---

## 11. How to Run Locally

1. Clone the repository: https://github.com/Unyime-Akan-Ibekwe/ai_10022200192.git
2. Install dependencies: pip install -r requirements.txt
3. Run the app: streamlit run ui.py

---

## 12. Limitations and Future Work

- The FAISS index is rebuilt from scratch on each cold start — persistent index caching to disk would improve startup time
- The election dataset covers only presidential results, not parliamentary
- Query expansion is rule-based — a learned expansion model would generalise better to unexpected query phrasings
- The domain boost keyword list is static — user feedback could be used to dynamically adjust weights over time
