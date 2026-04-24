import streamlit as st
from rag import load_csv, load_pdf, chunk_text, build_index, hybrid_retrieve
import traceback
import os
import requests

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Acity AI Assistant", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center;'>🏛️ Academic City RAG Assistant</h1>
    <p style='text-align: center; color: gray;'>
        Ask questions about Ghana Elections & 2025 Budget
    </p>
    """,
    unsafe_allow_html=True
)

st.write("System initializing...")

# -------------------------
# LOAD SYSTEM (CACHED)
# -------------------------
@st.cache_resource(show_spinner=False)
def setup_system():
    try:
        csv_data, winner_chunks = load_csv("data/Ghana_Election_Result.csv")
        pdf_data = load_pdf("data/budget.pdf")

        pdf_chunks = chunk_text(pdf_data)
        csv_chunks = csv_data

        chunks = []
        chunks.extend(csv_chunks)
        chunks.extend(pdf_chunks)
        chunks.extend(winner_chunks)

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
def manage_context_window(results, max_chars=6000):
    current_text = ""
    filtered = []

    for chunk, score in results:
        if len(current_text) + len(chunk) < max_chars:
            current_text += chunk + "\n"
            filtered.append((chunk, score))

    return filtered


GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

def generate_answer(prompt):
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Unexpected error: {str(e)}"


def build_prompt(query, results):
    context = "\n".join([r[0] for r in results])

    return f"""You are a helpful assistant answering questions about Ghana Elections and the 2025 Ghana Budget.

Use the context below to answer the question. The context contains election results and budget information.
If the answer is in the context, answer clearly and directly.
If the context is partially relevant, use what you can and say so.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""


# -------------------------
# USER INPUT
# -------------------------
query = st.text_input("🔎 Ask your question", "")

# -------------------------
# MAIN PIPELINE
# -------------------------
if st.button("🚀 Submit") and query:

    # STEP 1: RETRIEVAL
    results = hybrid_retrieve(query, chunks, index, k=12)

    st.subheader("🔍 Retrieved Chunks")
    for i, (chunk, score) in enumerate(results):
        st.markdown(
            f"""
**Chunk {i+1}**  
Score: {score:.4f}  

{chunk[:300]}...
"""
        )

    # STEP 2: CONTEXT FILTERING
    filtered_results = manage_context_window(results)

    # STEP 3: PROMPT BUILDING
    prompt = build_prompt(query, filtered_results)

    st.subheader("🧠 Final Prompt")
    st.code(prompt, language="text")

    # STEP 4: GENERATION
    with st.spinner("Generating answer..."):
        generated_text = generate_answer(prompt)

    # STEP 5: CLEAN OUTPUT
    cleaned = ". ".join(dict.fromkeys(generated_text.split(". ")))

    st.subheader("🤖 AI Response")
    st.markdown(
        f"""
<div style="
    background-color:#161B22;
    padding:15px;
    border-radius:10px;
    border:1px solid #30363d;
">
{cleaned}
</div>
""",
        unsafe_allow_html=True
    )
