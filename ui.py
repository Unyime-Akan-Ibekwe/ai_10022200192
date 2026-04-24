import streamlit as st
from rag import load_csv, load_pdf, chunk_text, build_index, hybrid_retrieve
import traceback
import requests

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Acity AI Assistant", layout="wide")

st.markdown("""
<h1 style='text-align: center;'>🏛️ Academic City RAG Assistant</h1>
<p style='text-align: center; color: gray;'>
Ask questions about Ghana Elections & 2025 Budget
</p>
""", unsafe_allow_html=True)

st.write("System initializing...")

# -------------------------
# LOAD SYSTEM
# -------------------------
@st.cache_resource(show_spinner=False)
def setup_system():
    try:
        csv_data, winner_chunks = load_csv("data/Ghana_Election_Result.csv")
        pdf_data = load_pdf("data/budget.pdf")

        pdf_chunks = chunk_text(pdf_data)
        chunks = csv_data + pdf_chunks + winner_chunks

        index = build_index(chunks)

        return chunks, index

    except Exception:
        st.error("System failed to load ❌")
        st.text(traceback.format_exc())
        st.stop()


chunks, index = setup_system()
st.success("System loaded successfully ✅")

# -------------------------
# HELPERS
# -------------------------
def manage_context_window(results, max_chars=2500):
    current_text = ""
    filtered = []

    for chunk, score in results:
        if len(current_text) + len(chunk) < max_chars:
            current_text += chunk + "\n"
            filtered.append((chunk, score))

    return filtered


def build_prompt(query, results):
    context = "\n".join([r[0] for r in results])

    return f"""
You are a precise academic assistant.

Answer ONLY using the context below.

Rules:
- Be clear and factual
- Use bullet points
- Combine information if needed
- If not found, say "Answer not found in context"

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""


# -------------------------
# HF API CALL
# -------------------------
def generate_answer(prompt):
    try:
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"

        headers = {
            "Authorization": f"Bearer {st.secrets['HF_API_TOKEN']}"
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            return f"Error {response.status_code}: {response.text}"

        result = response.json()

        if isinstance(result, list):
            return result[0].get("generated_text", "No output")

        if isinstance(result, dict):
            return result.get("generated_text", str(result))

        return str(result)

    except Exception as e:
        return f"Request failed: {str(e)}"


# -------------------------
# USER INPUT
# -------------------------
query = st.text_input("🔎 Ask your question")

# -------------------------
# MAIN PIPELINE
# -------------------------
if st.button("🚀 Submit") and query:

    # STEP 1: RETRIEVE
    results = hybrid_retrieve(query, chunks, index, k=8)

    st.subheader("🔍 Retrieved Chunks")

    for i, (chunk, score) in enumerate(results):
        st.markdown(f"""
**Chunk {i+1}**  
Score: `{score:.4f}`  

{chunk[:300]}...
""")

    # STEP 2: FILTER
    filtered_results = manage_context_window(results)

    # STEP 3: PROMPT
    prompt = build_prompt(query, filtered_results)

    st.subheader("🧠 Final Prompt")
    st.code(prompt)

    # STEP 4: GENERATE
    with st.spinner("Generating answer..."):
        generated_text = generate_answer(prompt)

    # STEP 5: CLEAN
    cleaned = ". ".join(dict.fromkeys(generated_text.split(". ")))

    st.subheader("🤖 AI Response")

    st.markdown(f"""
<div style="
background-color:#161B22;
padding:15px;
border-radius:10px;
border:1px solid #30363d;">
{cleaned}
</div>
""", unsafe_allow_html=True)
