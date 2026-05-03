import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import faiss
import numpy as np

# ---------------- API KEY ----------------
genai.configure(api_key="YOUR_GEMINI_API_KEY")

# ---------------- UI ----------------
st.title("📘 AI Lab Manual Assistant")
st.write("Upload PDFs and ask questions!")

uploaded_files = st.file_uploader(
    "Upload Lab Manuals",
    type="pdf",
    accept_multiple_files=True
)

query = st.text_input("Ask a question from the lab manual:")

# ---------------- PROCESS PDFs ----------------
def process_pdfs(files):
    documents = []
    
    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                documents.append(text)

    # Split into chunks
    chunks = []
    chunk_size = 500

    for doc in documents:
        for i in range(0, len(doc), chunk_size):
            chunks.append(doc[i:i+chunk_size])

    return chunks

# ---------------- CREATE VECTOR STORE ----------------
def create_vector_store(chunks):
    embeddings = []

    for chunk in chunks:
        emb = genai.embed_content(
            model="models/embedding-001",
            content=chunk
        )
        embeddings.append(emb["embedding"])

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    return index, chunks

# ---------------- GET ANSWER ----------------
def get_answer(query, index, chunks):
    query_embedding = genai.embed_content(
        model="models/embedding-001",
        content=query
    )["embedding"]

    query_embedding = np.array([query_embedding]).astype("float32")

    D, I = index.search(query_embedding, k=3)

    context = " ".join([chunks[i] for i in I[0]])

    model = genai.GenerativeModel("gemini-pro")

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = model.generate_content(prompt)

    return response.text

# ---------------- MAIN LOGIC ----------------
if uploaded_files and query:
    with st.spinner("Processing..."):
        chunks = process_pdfs(uploaded_files)
        index, chunks = create_vector_store(chunks)
        answer = get_answer(query, index, chunks)

    st.subheader("Answer:")
    st.write(answer)
