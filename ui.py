import streamlit as st
from rag import load_csv, load_pdf, chunk_text, build_index, hybrid_retrieve
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import traceback

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Acity AI Assistant", layout="wide")

st.markdown("""
<h1 style='text-align: center;'>🏛️ Academic City RAG Assistant</h1>
<p style='text-align: center; color: gray;'>Ask questions about Ghana Elections & 2025 Budget</p>
""", unsafe_allow_html=True)

st.write("System initializing...")

# -------------------------
# LOAD SYSTEM (CACHED)
# -------------------------
@st.cache_resource(show_spinner=False)
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

        model_id = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_llm = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        return chunks, index, tokenizer, model_llm

    except Exception as e:
        st.error("System failed to load")
        st.text(traceback.format_exc())
        st.stop()


chunks, index, tokenizer, model_llm = setup_system()

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
# USER INPUT
# -------------------------
query = st.text_input("🔎 Ask your question", "")

# -------------------------
# MAIN PIPELINE
# -------------------------
if st.button("🚀 Submit") and query:

    # STEP 1: RETRIEVAL
    results = hybrid_retrieve(query, chunks, index, k=8)

    st.subheader("🔍 Retrieved Chunks")

    for i, (chunk, score) in enumerate(results):
        st.markdown(f"""
        **Chunk {i+1}**  
        Score: `{score:.4f}`  
        {chunk[:300]}...
        """)

    # STEP 2: CONTEXT FILTERING
    filtered_results = manage_context_window(results)

    # STEP 3: PROMPT BUILDING
    prompt = build_prompt(query, filtered_results)

    st.subheader("🧠 Final Prompt")
    st.code(prompt, language="text")

    # STEP 4: GENERATION
    with st.spinner("Generating answer..."):

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # STEP 5: OUTPUT CLEANING (simple)
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
