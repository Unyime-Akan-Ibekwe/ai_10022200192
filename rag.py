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
    df["Votes"] = pd.to_numeric(df["Votes"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
    df["Votes(%)"] = pd.to_numeric(df["Votes(%)"].astype(str).str.replace("%", ""), errors="coerce").fillna(0)

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
    if "won" in query or "winner" in query or "election" in query:
        expansions.append(query + " national winner total votes nationwide result")

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

        # Domain-specific scoring function (Innovation Component)
        domain_boost = 0

        # Budget domain keywords
        if any(word in chunk_lower for word in ["risk", "challenge", "threat", "debt", "inflation"]):
            domain_boost += 0.3
        if any(word in chunk_lower for word in ["budget", "fiscal", "revenue", "expenditure", "gdp", "deficit"]):
            domain_boost += 0.3
        if any(word in chunk_lower for word in ["petroleum", "oil", "energy", "cocoa", "gold"]):
            domain_boost += 0.2

        # Election domain keywords
        if any(word in chunk_lower for word in ["winner", "national", "votes", "candidate", "party"]):
            domain_boost += 0.3
        if any(word in chunk_lower for word in ["ndc", "npp", "electoral", "region", "election"]):
            domain_boost += 0.2

        # Boost chunks that contain numbers/statistics (more factual)
        import re
        if re.search(r'\d+', chunk_lower):
            domain_boost += 0.1

        # Cap boost to avoid overwhelming vector score
        domain_boost = min(domain_boost, 1.0)

        final_score = (0.5 * vector_score) + (0.3 * keyword_score) + (0.5 * domain_boost)

        scored.append((chunk, final_score))

    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    return scored[:k]


def get_winners_from_csv(df):
    winners = []

    # National winner per year (sum votes across all regions)
    grouped_year = df.groupby(["Year", "Candidate", "Party"])["Votes"].sum().reset_index()

    for year, group in grouped_year.groupby("Year"):
        national_winner = group.loc[group["Votes"].idxmax()]
        total_votes = group["Votes"].sum()
        pct = round((national_winner["Votes"] / total_votes) * 100, 2)

        winners.append(
            f"NATIONAL ELECTION RESULT: National winner of {year} Ghana election: {national_winner['Candidate']} "
            f"National winner of {year} Ghana election: {national_winner['Candidate']} "
            f"({national_winner['Party']}) with {national_winner['Votes']:,} total votes nationwide ({pct}%)"
        )

    # Regional winners per year
    for (year, region), group in df.groupby(["Year", "New Region"]):
        regional_winner = group.loc[group["Votes"].idxmax()]
        winners.append(
            f"In {year}, {region} region winner: {regional_winner['Candidate']} "
            f"({regional_winner['Party']}) with {regional_winner['Votes']:,} votes"
        )

    return winners
