import os
import shutil
import torch
import streamlit as st
from pathlib import Path
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
from PyPDF2 import PdfReader
import re

# Initialize Pinecone
api_key = st.secrets["pinecone"]["api_key"]
env = st.secrets["pinecone"]["ENV"]
index_name = st.secrets["pinecone"]["INDEX_NAME"]
hf_token = st.secrets["huggingface"]["token"]

pc = Pinecone(api_key=api_key, environment=env)
index = pc.Index(index_name)

# Load Transformer Model for Embeddings
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModel.from_pretrained(model_name, use_auth_token=hf_token)

storage_folder = "content_storage"
os.makedirs(storage_folder, exist_ok=True)

# Define important sections with headers
SECTION_HEADERS = [
    "Scope of Work", "Eligibility Criteria", "Bid Submission Date", "Technical Requirements",
    "Evaluation Criteria", "Financial Bid", "Pre-bid Meeting", "Security Deposit", "EMD Amount"
]

def extract_important_sections(text):
    """Extracts structured content from the PDF based on defined section headers."""
    extracted_data = {}
    
    for header in SECTION_HEADERS:
        pattern = rf"{header}.*?(?=\n[A-Z].*?:|\Z)"  # Extract the section until the next major title
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            extracted_data[header] = match.group(0).strip()

    return extracted_data

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    reader = PdfReader(file_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    return extract_important_sections(text) if text.strip() else {}

def embed_text(text):
    """Generates an embedding for the given text chunk."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        return model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

def store_in_pinecone(file_name, structured_content):
    """Stores extracted structured content in Pinecone with embeddings."""
    for section, content in structured_content.items():
        chunk_id = f"{file_name}_{section}"
        embedding = embed_text(content)
        metadata = {"file_name": file_name, "section": section, "text": content}

        if len(embedding) == 768:
            index.upsert([(chunk_id, embedding, metadata)])
            st.write(f"✅ Stored {section} from {file_name} in Pinecone.")
        else:
            st.error(f"❌ Invalid vector size: Expected 768, got {len(embedding)}.")

def process_folder(folder_path):
    """Processes all PDFs in a folder and stores structured content in Pinecone."""
    st.write(f"📂 Processing folder: {folder_path}")

    for file in Path(folder_path).rglob("*.pdf"):
        st.write(f"📄 Processing {file.name}...")
        structured_content = extract_text_from_pdf(file)

        if structured_content:
            store_in_pinecone(file.name, structured_content)
        else:
            st.warning(f"⚠️ No important content found in {file.name}.")

def main():
    st.title("📌 Process & Store Structured Tender Data in Pinecone")

    uploaded_zip = st.file_uploader("📁 Upload a ZIP folder containing PDFs", type=["zip"])

    if uploaded_zip:
        zip_path = os.path.join(storage_folder, uploaded_zip.name)
        
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getvalue())

        folder_extract_path = os.path.join(storage_folder, "extracted_files")
        shutil.unpack_archive(zip_path, folder_extract_path)
        st.write(f"📂 Folder extracted to {folder_extract_path}")

        process_folder(folder_extract_path)
        st.success("✅ All important structured data stored in Pinecone!")

if __name__ == "__main__":
    main()
