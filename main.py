import os
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
from PyPDF2 import PdfReader
from docx import Document
import torch

# Initialize Pinecone
api_key = st.secrets["pinecone"]["api_key"]
env = st.secrets["pinecone"]["ENV"]
index_name = st.secrets["pinecone"]["INDEX_NAME"]
hf_token = st.secrets["huggingface"]["token"]

pc = Pinecone(api_key=api_key, environment=env)
index = pc.Index(index_name)

# Load model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModel.from_pretrained(model_name, use_auth_token=hf_token)

storage_folder = "content_storage"
os.makedirs(storage_folder, exist_ok=True)

# Function to chunk text
def chunk_text(text, chunk_size=512, overlap=50):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks

# Function to extract text from PDFs
def process_pdf(file_path):
    reader = PdfReader(file_path)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return chunk_text(text) if text.strip() else []

# Function to extract text from DOCX
def process_docx(file_path):
    doc = Document(file_path)
    text = " ".join([para.text for para in doc.paragraphs if para.text.strip()])
    return chunk_text(text) if text.strip() else []

# Function to extract text from TXT files
def process_text(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        text = file.read()
    return chunk_text(text) if text.strip() else []

# Function to generate embeddings
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        return model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

# Function to store chunks in Pinecone
def store_chunks(file_name, chunks):
    for i, chunk in enumerate(chunks):
        chunk_id = f"{file_name}_chunk_{i}"
        embedding = embed_text(chunk)
        metadata = {"file_name": file_name, "chunk_id": i, "text": chunk}
        
        if len(embedding) == 768:  # Ensure correct vector size
            index.upsert([(chunk_id, embedding, metadata)])
            st.write(f"✅ Stored {file_name} - chunk {i} in Pinecone.")
        else:
            st.error(f"❌ Invalid vector size: Expected 768, got {len(embedding)}.")

# Function to retrieve stored data
def retrieve_data():
    st.subheader("📂 View Stored Data in Pinecone")
    response = index.describe_index_stats()
    
    if "namespaces" in response and response["namespaces"]:
        st.write(f"Total Vectors Stored: {response['total_vector_count']}")
        query = st.text_input("🔍 Search (Leave blank to view all)")
        
        if st.button("Search"):
            results = index.query(top_k=10, include_metadata=True)
            if results["matches"]:
                for match in results["matches"]:
                    metadata = match["metadata"]
                    st.write(f"**📌 Chunk {metadata['chunk_id']} from {metadata['file_name']}**")
                    st.write(f"📄 {metadata['text']}")
                    st.write("---")
            else:
                st.warning("❌ No matching results found.")
    else:
        st.warning("No data found in Pinecone.")

# Streamlit UI
st.title("📌 Tender Content Storage & Retrieval")

uploaded_file = st.file_uploader("📁 Upload Tender (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    file_path = os.path.join(storage_folder, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.write(f"📂 Processing {uploaded_file.name}...")

    chunks = []
    if uploaded_file.name.endswith(".pdf"):
        chunks = process_pdf(file_path)
    elif uploaded_file.name.endswith(".docx"):
        chunks = process_docx(file_path)
    elif uploaded_file.name.endswith(".txt"):
        chunks = process_text(file_path)

    if chunks:
        store_chunks(uploaded_file.name, chunks)
        st.success(f"✅ {len(chunks)} chunks stored successfully!")
    else:
        st.warning("❌ No content extracted from document.")

retrieve_data()  # Show stored data in Pinecone
