# STEP 2: FIRST REAL CODE (DATA LOADING)

import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Loading the CSV file
def load_csv(path):
    df = pd.read_csv(path)
    df = df.fillna("")
    print(df.columns)
    
    # Converting rows to text
    texts = df.apply(lambda row: 
    f"Year {row['Year']}, Region {row['New Region']}, "
    f"Candidate {row['Candidate']}, Party {row['Party']}, "
    f"Votes {row['Votes']}, Percentage {row['Votes(%)']}", 
    axis=1)
    winners = get_winners_from_csv(df)

    return texts.tolist(), winners

# Loading the PDF file
def load_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def detect_query_type(query):
    query = query.lower()

    if "who won" in query or "winner" in query:
        return "winner"
    
    return "normal"

# Text chunking - CHUNKING
def chunk_text(text, chunk_size=500, overlap=100):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks




model = SentenceTransformer('all-MiniLM-L6-v2')

def build_index(chunks):
    embeddings = model.encode(chunks, normalize_embeddings=True)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# STEP 5: ADD EMBEDDINGS + FAISS
# STEP 8: MODIFY RETRIEVAL
def retrieve(query, chunks, index, k=3):
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")

    scores, indices = index.search(q_emb, k)

    results = []

    for score, i in zip(scores[0], indices[0]):
        results.append((chunks[i], float(score)))

    return results



def expand_query(query):
    query = query.lower()

    expansions = [query]

    if "risk" in query:
        expansions.append(query + " challenges problems threats")
    if "inflation" in query:
        expansions.append(query + " price increase cost of living")
    if "growth" in query:
        expansions.append(query + " gdp economy expansion")
    if "budget" in query:
        expansions.append(query + " fiscal policy government spending")
    if "fiscal" in query:
        expansions.append(query + " debt arrears deficit risks challenges energy cocoa financial sector")

    return expansions


def hybrid_retrieve(query, chunks, index, k=5):
    expanded_queries = expand_query(query)

    all_results = []

    for q in expanded_queries:
        results = retrieve(q, chunks, index, k*2)
        all_results.extend(results)

    STOPWORDS = {"who", "what", "is", "the", "a", "of", "in", "to", "which", "on", "for", "and", "with", "by", "as", "from", "that", "this", "are", "was", "were"}

    keywords = [w for w in query.lower().split() if w not in STOPWORDS]

    scored = []

    for chunk, vector_score in all_results:
        chunk_lower = chunk.lower()

        keyword_score = sum(1 for word in keywords if word in chunk_lower)
        keyword_score = keyword_score / len(keywords) if keywords else 0

        # smarter domain boosting
        domain_boost = 0
        if any(word in chunk_lower for word in ["risk", "challenge", "threat", "debt", "inflation"]):
            domain_boost += 0.5

        final_score = (0.5 * vector_score) + (0.3 * keyword_score) + (0.5 * domain_boost)

        scored.append((chunk, final_score))

    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    return scored[:k]



def get_winners_from_csv(df):
    winners = []

    grouped = df.groupby("Year")

    for year, group in grouped:
        winner = group.loc[group["Votes"].idxmax()]

        winners.append(
            f"Year {year} winner: {winner['Candidate']} ({winner['Party']}) with {winner['Votes']} votes ({winner['Votes(%)']}%)"
        )

    return winners

