import streamlit as st
from rag import load_csv, load_pdf, chunk_text, build_index, hybrid_retrieve
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import traceback



# 1. Page Config MUST be the very first Streamlit command
st.set_page_config(page_title="Acity AI Assistant", layout="wide")

# 2. Now you can use markdown and other UI element
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center;'>🏛️ Academic City RAG Assistant</h1>
<p style='text-align: center; color: gray;'>Ask questions about Ghana Elections & 2025 Budget</p>
""", unsafe_allow_html=True)

# --- 1. CACHED SETUP (Runs once) ---
@st.cache_resource
def setup_system():
    try:
        with st.spinner("Loading Data & Models... (This takes a moment)"):
          # Data
          csv_data, winner_chunks = load_csv("data/Ghana_Election_Result.csv")
          pdf_data = load_pdf("data/budget.pdf")
        
          pdf_chunks = chunk_text(pdf_data)
          csv_chunks = csv_data
        
          chunks = []
          chunks.extend(csv_chunks)
          chunks.extend(pdf_chunks)
          chunks.extend(winner_chunks)
        
          # Index
          index = build_index(chunks)
        
          # LLM
          model_id = "google/flan-t5-small"
          tokenizer = AutoTokenizer.from_pretrained(model_id)
          model_llm = AutoModelForSeq2SeqLM.from_pretrained(model_id)

          st.write("FILES:", os.listdir("data"))
        
          return chunks, index, tokenizer, model_llm

    except Exception as e:
        st.error("CRASH DETECTED:")
        st.text(str(e))
        st.text(traceback.format_exc())
        raise e


st.write("Data loaded successfully")

st.write(os.listdir("data"))

chunks, index, tokenizer, model_llm = setup_system()

# --- 2. HELPER FUNCTIONS ---
def manage_context_window(results, max_chars=2000):
    current_text = ""
    filtered_results = []
    for chunk, score in results:
        if len(current_text) + len(chunk) < max_chars:
            current_text += chunk + "\n"
            filtered_results.append((chunk, score))
    return filtered_results

def build_prompt(query, results):
    context = "\n".join([r[0] for r in results])

    return f"""
You are a precise academic assistant.

Your task is to extract factual answers ONLY from the provided context.

STRICT RULES:
- Do NOT repeat sentences
- Do NOT invent information
- Do NOT restate the question
- Answer in clear bullet points
- Be concise and direct
- If the answer is not found, say: "Answer not found in context"

CONTEXT:
{context}

QUESTION:
{query}

FINAL ANSWER (bullet points only):
"""




# --- 3. USER INTERFACE ---
with st.container():
    query = st.text_input(
        "🔎 Ask your question",
        placeholder="e.g. Who won the 2020 election?"
    )


if st.button("🚀 Submit", use_container_width=True):
    if query:
        # Retrieval
        st.subheader("🔍 Retrieval Stage")
        results = hybrid_retrieve(query, chunks, index, k=20)

        print("\n--- DEBUG RETRIEVAL ---")
        for chunk, score in results:
            print(f"SCORE: {score}")
            print(chunk[:300])
            print("-"*50)

        
        # UI Requirement: Display Retrieved Chunks
        with st.expander("🔍 Retrieved Context"):
            for i, (chunk, score) in enumerate(results):
              st.markdown(f"""
              **Chunk {i+1}**  
              Score: `{score:.4f}`  
              {chunk[:250]}...
               """)

            
        # UI Requirement: Show final response

        with st.spinner("Thinking... generating answer..."):

            # Context Management & Prompting
            filtered_results = manage_context_window(results, max_chars=3000)
            prompt = build_prompt(query, filtered_results)
            
            # Manual LLM Generation
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model_llm.generate(
                                    **inputs,
                                    max_new_tokens=150,
                                    do_sample=False,   
                                    repetition_penalty=1.2,  # stops looping
                                    length_penalty=1.0
)


            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 🔥 Remove repeated phrases
            lines = generated_text.split(". ")
            cleaned = []
            for line in lines:
                if line not in cleaned:
                    cleaned.append(line)

            generated_text = ". ".join(cleaned)

            
            st.markdown("### 🤖 AI Response")
            st.markdown(f"""
            <div style="
                background-color:#161B22;
                padding:15px;
                border-radius:10px;
                border:1px solid #30363d;">
            {generated_text}
            </div>
            """, unsafe_allow_html=True)

            
            # Optional: Show the prompt in an expander so the grader can see it
            with st.expander("View Final Prompt Sent to LLM"):
                st.code(prompt, language="text")
