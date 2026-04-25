# AI RAG Chatbot - Academic City
Name: Unyimeabasi Akan Ibekwe
Index Number: 10022200192

## Project Description
This project implements a Retrieval-Augmented Generation (RAG) system that allows users to query Ghana Election data and the 2025 Ghana Budget Statement. The system was built entirely from scratch without end-to-end frameworks like LangChain or LlamaIndex. All core components including chunking, embedding, retrieval, and prompt construction were implemented manually.

## Features
- Query input
- Query expansion for improved retrieval
- Hybrid retrieval (keyword + vector search)
- Domain-specific scoring function
- Retrieval of relevant chunks with similarity scores
- Context window management
- AI-generated responses via Groq API

## Technologies Used
- Python
- FAISS
- Sentence Transformers (all-MiniLM-L6-v2)
- Streamlit
- Groq API (llama-3.3-70b-versatile)
- pdfplumber
- pandas
- numpy

## How to Run
```bash
streamlit run ui.py
```

## Deployed App
https://unyime-akan-ibekwe-ai-10022200192-ui-m00nyv.streamlit.app
