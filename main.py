import streamlit as st
import fitz  
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import time
import re

# Load secrets
PINECONE_API_KEY = st.secrets["pinecone"]["api_key"]
PINECONE_ENV = st.secrets["pinecone"]["env"]
PINECONE_INDEX = st.secrets["pinecone"]["index_name"]
HF_TOKEN = st.secrets["huggingface"]["token"]

os.environ['HUGGINGFACE_API_KEY'] = HF_TOKEN

# Pinecone Setup
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(name=PINECONE_INDEX, dimension=384, metric='cosine')
index = pc.Index(PINECONE_INDEX)

# Sections to Extract
SECTIONS = ["Introduction", "Project Details", "Bidding Process", "Scope of Work", "Payment Terms"]

def extract_sections(text):
    """Extract key sections from the PDF text"""
    extracted_data = {}
    for i in range(len(SECTIONS) - 1):
        start_pattern = rf"{SECTIONS[i]}"
        end_pattern = rf"{SECTIONS[i + 1]}"
        match = re.search(f"{start_pattern}(.*?){end_pattern}", text, re.DOTALL)
        if match:
            extracted_data[SECTIONS[i]] = match.group(1).strip()
    
    # Extract last section (Payment Terms)
    last_section = re.search(rf"{SECTIONS[-1]}(.*)", text, re.DOTALL)
    if last_section:
        extracted_data[SECTIONS[-1]] = last_section.group(1).strip()

    return extracted_data

class PDFLoader:
    def __init__(self, pdf_file):
        if pdf_file is None:
            raise ValueError("PDF file is not provided.")
        self.pdf_file = pdf_file
        self.extracted_text = self.extract_text()

        # Extract structured sections
        self.structured_data = extract_sections(self.extracted_text)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        self.text_chunks = text_splitter.split_text(self.extracted_text)

    def extract_text(self):
        """Extract text from PDF"""
        doc = fitz.open(stream=self.pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, text_chunks):
        return self.model.encode(text_chunks)

def store_embeddings(index, embeddings, metadata):
    """Upsert embeddings into Pinecone"""
    upsert_data = [{"id": f'doc-{i}', "values": embeddings[i].tolist(), "metadata": metadata[i]} for i in range(len(embeddings))]

    try:
        response = index.upsert(vectors=upsert_data)
        st.success(f"Successfully upserted {len(embeddings)} vectors.")
    except Exception as e:
        st.error(f"Error during Pinecone upsert: {str(e)}")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Process PDF
    loader = PDFLoader(uploaded_file)
    text_chunks = loader.text_chunks
    structured_data = loader.structured_data

    # Generate embeddings
    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.generate_embeddings(text_chunks)

    # Store embeddings
    metadata = [{"chunk_index": i, "source": "uploaded_pdf"} for i in range(len(embeddings))]
    store_embeddings(index, embeddings, metadata)

    # Display extracted structured sections
    st.subheader("Extracted Sections")
    for section, content in structured_data.items():
        with st.expander(section):
            st.write(content[:2000])  # Limit display to first 2000 chars
