import os
import shutil
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
import torch
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document

# Load secrets from Streamlit
api_key = st.secrets["pinecone"]["api_key"]
env = st.secrets["pinecone"]["ENV"]
index_name = st.secrets["pinecone"]["INDEX_NAME"]
hf_token = st.secrets["huggingface"]["token"]

# Initialize Pinecone instance
pc = Pinecone(api_key=api_key, environment=env)

model_name = "distilbert-base-uncased"  # A smaller, easier model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModel.from_pretrained(model_name, use_auth_token=hf_token)

# Initialize Pinecone index
index = pc.Index(index_name)  # This is where the index is initialized

def process_text_file(file_path):
    """Process text files and extract embeddings and metadata"""
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()

        inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

        # Example metadata for text files
        metadata = {
            "file_name": file_path.name,
            "file_type": "text",
            "introduction": "This is the introduction section of the document.",  # Example metadata
            "scope_of_work": "The scope of work involves analyzing and processing data.",  # Example metadata
            "content_preview": content[:200]  # Preview of the content
        }
        
        return embeddings, metadata
    except Exception as e:
        st.error(f"Error processing text file {file_path}: {e}")
        return None, None

def process_pdf_file(file_path):
    """Extract text from PDF and generate embeddings with metadata"""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            content = ''
            for page in reader.pages:
                content += page.extract_text()

        if content.strip():  # If there's text content in the PDF
            inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

            # Example metadata for PDF files
            metadata = {
                "file_name": file_path.name,
                "file_type": "pdf",
                "introduction": "This document provides an overview of the tender process.",  # Example metadata
                "scope_of_work": "The scope includes the analysis and processing of tender data.",  # Example metadata
                "content_preview": content[:200]  # Preview of the content
            }
            
            return embeddings, metadata
        else:
            return None, None
    except Exception as e:
        st.error(f"Error processing PDF file {file_path}: {e}")
        return None, None

def process_docx_file(file_path):
    """Extract text from DOCX and generate embeddings with metadata"""
    try:
        doc = Document(file_path)
        content = ''
        for para in doc.paragraphs:
            content += para.text

        if content.strip():  # If there's text content in the DOCX
            inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

            # Example metadata for DOCX files
            metadata = {
                "file_name": file_path.name,
                "file_type": "docx",
                "introduction": "This document contains tender housekeeping details.",  # Example metadata
                "scope_of_work": "The scope of work described in the tender includes housekeeping services.",  # Example metadata
                "content_preview": content[:200]  # Preview of the content
            }
            
            return embeddings, metadata
        else:
            return None, None
    except Exception as e:
        st.error(f"Error processing DOCX file {file_path}: {e}")
        return None, None

def store_in_pinecone(file_name, file_content, metadata):
    """Store the embeddings and metadata in Pinecone"""
    try:
        if file_content:
            vector = file_content
            if len(vector) == 768:  # Ensure correct embedding dimension
                # Storing both vector and metadata in Pinecone
                index.upsert([(file_name, vector, metadata)])
                st.write(f"Stored {file_name} with metadata in Pinecone.")
            else:
                st.error(f"Invalid vector dimension for {file_name}. Expected 768, got {len(vector)}.")
        else:
            st.warning(f"No content to store for {file_name}. Skipping.")
    except Exception as e:
        st.error(f"Error storing {file_name} in Pinecone: {e}")

def query_pinecone(query_vector):
    """Query Pinecone and display results"""
    try:
        result = index.query(queries=[query_vector], top_k=5)
        if result.matches:
            for match in result.matches:
                st.write(f"ID: {match.id}")
                st.write(f"Score: {match.score}")
                st.write(f"Metadata: {match.metadata}")  # Showing full metadata
        else:
            st.warning("No matches found in Pinecone.")
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")

def main():
    """Main function to handle file upload and processing"""
    st.title("Folder Upload and Pinecone Storage")

    uploaded_folder = st.file_uploader("Choose a folder to upload", type=["zip"], accept_multiple_files=False)

    if uploaded_folder is not None:
        save_path = Path("local_upload_folder")
        
        if save_path.exists():
            shutil.rmtree(save_path)  # Clean up existing folder
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / uploaded_folder.name, "wb") as f:
            f.write(uploaded_folder.getvalue())
        
        try:
            shutil.unpack_archive(save_path / uploaded_folder.name, save_path)
            st.write(f"Folder uploaded and unzipped to {save_path}. Processing files...")
        except Exception as e:
            st.error(f"Error extracting ZIP file: {e}")
            return

        # Process each file in the folder
        for file in Path(save_path).rglob('*.*'):
            if file.is_file():
                st.write(f"Processing {file.name}...")
                try:
                    file_content = None
                    metadata = None
                    if file.suffix == '.txt':  # For text files
                        file_content, metadata = process_text_file(file)
                    elif file.suffix == '.pdf':  # For PDF files
                        file_content, metadata = process_pdf_file(file)
                    elif file.suffix == '.docx':  # For DOCX files
                        file_content, metadata = process_docx_file(file)

                    if file_content:
                        store_in_pinecone(file.stem, file_content, metadata)
                    else:
                        st.warning(f"No text found in {file.name}. Skipping.")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")

        st.success("Folder contents stored in Pinecone.")

if __name__ == "__main__":
    main()
