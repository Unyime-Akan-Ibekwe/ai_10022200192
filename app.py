from rag import load_csv, load_pdf, chunk_text, build_index, retrieve
from rag import hybrid_retrieve
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

experiment_logs = []

def log_step(stage, data):
    log_entry = f"STAGE: {stage}\nDATA: {data}\n{'-'*30}"
    experiment_logs.append(log_entry)
    print(log_entry)


csv_data, winner_chunks = load_csv("data/Ghana_Election_Result.csv")
pdf_data = load_pdf("data/budget.pdf")

pdf_chunks = chunk_text(pdf_data)

csv_chunks = csv_data  # keep each row as one chunk

model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_llm = AutoModelForSeq2SeqLM.from_pretrained(model_id)



# Combine properly
chunks = []

chunks.extend(csv_chunks)
chunks.extend(pdf_chunks)
chunks.extend(winner_chunks)


# Build index
index = build_index(chunks)

# Test query
query = "Who won the election?"

# During retrieval
results = hybrid_retrieve(query, chunks, index, k=3)
log_step("Retrieval", f"Found {len(results)} chunks for query: {query}")

def manage_context_window(results, max_chars=1000):
    current_text = ""
    filtered_results = []
    for chunk, score in results:
        if len(current_text) + len(chunk) < max_chars:
            current_text += chunk + "\n"
            filtered_results.append((chunk, score))
    return filtered_results


print("\n--- RETRIEVAL LOG ---")
for chunk, score in results:
    print(f"Score: {score:.4f}")
    print(f"Chunk: {chunk[:200]}")
    print("-" * 50)


print("\n--- FINAL QUERY ---")
print(query)


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




# Use it before building the prompt:
filtered_results = manage_context_window(results)
prompt = build_prompt(query, filtered_results)


# During prompting
prompt = build_prompt(query, results)
log_step("Prompt Construction", prompt)

print("\n--- FINAL PROMPT ---")
print(prompt)


print("\n--- LLM RESPONSE ---")
# Manually encode the prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate response manually (Manual Implementation - Exam Part D)
outputs = model_llm.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=False,   
    repetition_penalty=1.2,  # stops looping
    length_penalty=1.0
)



# Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Remove repeated phrases
lines = generated_text.split(". ")
cleaned = []
for line in lines:
    if line not in cleaned:
        cleaned.append(line)

generated_text = ". ".join(cleaned)


print(generated_text)

# Log the final output
log_step("Final LLM Response", generated_text)


print("\n--- PART B: FAILURE CASE EXPERIMENT ---")
bad_query = "What is the recipe for Jollof rice?"
bad_results = hybrid_retrieve(bad_query, chunks, index, k=3)

print(f"Query: {bad_query}")
for chunk, score in bad_results:
    # This shows the system retrieving irrelevant budget/election info for a food query
    print(f"Retrieved Irrelevant Chunk (Score {score:.4f}): {chunk[:100]}...")

print("\nFIX IMPLEMENTED: Similarity Thresholding")
# The fix: Only accept results with a score above a certain threshold
threshold = 0.5
fixed_results = [r for r in bad_results if r[1] > threshold]

if not fixed_results:
    print("System correctly identified no relevant context found (Fix Active).")