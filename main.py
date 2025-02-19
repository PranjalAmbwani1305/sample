import os
import shutil
import torch
import streamlit as st
from pathlib import Path
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
from PyPDF2 import PdfReader
import re

api_key = st.secrets["pinecone"]["api_key"]
env = st.secrets["pinecone"]["ENV"]
index_name = st.secrets["pinecone"]["INDEX_NAME"]
hf_token = st.secrets["huggingface"]["token"]

pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModel.from_pretrained(model_name, token=hf_token)

storage_folder = "content_storage"
os.makedirs(storage_folder, exist_ok=True)

def extract_text_sections(text):
    section_keywords = {
        "Introduction": r"(?i)\bIntroduction\b",
        "Project Detail": r"(?i)\bProject Details?\b",
        "Delivery Detail": r"(?i)\bDelivery Details?\b",
        "Bidding Detail": r"(?i)\bBidding Details?\b|\bTender Details?\b",
        "Payment Detail": r"(?i)\bPayment Terms?\b|\bPayment Details?\b",
        "Penalties Detail": r"(?i)\bPenalt(y|ies) Details?\b|\bPenalty Clause\b",
        "Scope of Work": r"(?i)\bScope of Work\b|\bWork Description\b"
    }
    
    sections = {}
    current_section = "Uncategorized"
    sections[current_section] = []

    for line in text.split("\n"):
        line = line.strip()
        for section_name, pattern in section_keywords.items():
            if re.search(pattern, line):
                current_section = section_name
                sections[current_section] = []
                break
        sections[current_section].append(line)
    
    return {k: "\n".join(v) for k, v in sections.items() if v}

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return extract_text_sections(full_text) if full_text.strip() else {}

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        return model(**inputs).last_hidden_state[:, 0, :].squeeze().tolist()

def store_in_pinecone(file_name, structured_content):
    for section, text in structured_content.items():
        chunk_id = f"{file_name}_{section.replace(' ', '_')}"
        embedding = embed_text(text)
        metadata = {"file_name": file_name, "section": section, "text": text}
        if len(embedding) == 768:
            index.upsert([(chunk_id, embedding, metadata)])
            st.write(f"✅ Stored {file_name} - {section} in Pinecone.")
        else:
            st.error(f"❌ Invalid vector size: Expected 768, got {len(embedding)}.")

def process_folder(folder_path):
    st.write(f"📂 Processing folder: {folder_path}")
    for file in Path(folder_path).rglob("*.pdf"):
        st.write(f"📄 Processing {file.name}...")
        structured_content = extract_text_from_pdf(file)
        if structured_content:
            store_in_pinecone(file.name, structured_content)
        else:
            st.warning(f"⚠️ No structured content found in {file.name}.")

def main():
    st.title("📌 Process & Store Structured PDF Data in Pinecone")
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
