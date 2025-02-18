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

# Initialize transformer model
model_name = "distilbert-base-uncased"  # A smaller, easier model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModel.from_pretrained(model_name, use_auth_token=hf_token)

def process_text_file(file_path):
    """Process text files and extract embeddings"""
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()

        inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

        return content, embeddings
    except Exception as e:
        st.error(f"Error processing text file {file_path}: {e}")
        return None, None

def process_pdf_file(file_path):
    """Extract text from PDF and generate embeddings"""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            content = ''
            for page in reader.pages:
                content += page.extract_text()
        
        if content.strip():  # If there's any text content in the PDF
            inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
            return content, embeddings
        else:
            return None, None
    except Exception as e:
        st.error(f"Error processing PDF file {file_path}: {e}")
        return None, None

def process_docx_file(file_path):
    """Extract text from DOCX and generate embeddings"""
    try:
        doc = Document(file_path)
        content = ''
        for para in doc.paragraphs:
            content += para.text

        if content.strip():  # If there's any text content in the DOCX
            inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
            return content, embeddings
        else:
            return None, None
    except Exception as e:
        st.error(f"Error processing DOCX file {file_path}: {e}")
        return None, None

def store_in_pinecone(file_name, file_content, embeddings, file_type):
    """Store the embeddings in Pinecone with all fields"""
    try:
        if file_content and embeddings:
            # Truncate content to avoid metadata size limit issues
            truncated_content = file_content[:2000]  # Truncate to the first 2000 characters
            vector = embeddings
            
            if len(vector) == 768:
                index = pc.Index(index_name)
                metadata = {
                    "file_name": file_name,
                    "file_type": file_type,
                    "content_preview": truncated_content,  # Store truncated content preview
                    "full_content": file_content  # Full content to be stored in metadata
                }
                index.upsert([(file_name, vector, metadata)])
                st.write(f"Stored {file_name} in Pinecone.")
            else:
                st.error(f"Invalid vector dimension for {file_name}. Expected 768, got {len(vector)}.")
        else:
            st.warning(f"No content to store for {file_name}. Skipping.")
    except Exception as e:
        st.error(f"Error storing {file_name} in Pinecone: {e}")

def main():
    st.title("Folder Upload and Pinecone Storage")

    uploaded_folder = st.file_uploader("Choose a folder to upload", type=["zip"], accept_multiple_files=False)

    if uploaded_folder is not None:
        save_path = Path("local_upload_folder")
        
        if save_path.exists():
            shutil.rmtree(save_path)  # Clean up existing folder
        save_path.mkdir(parents=True, exist_ok=True)

        # Save the uploaded zip folder
        with open(save_path / uploaded_folder.name, "wb") as f:
            f.write(uploaded_folder.getvalue())
        
        # Unzip the uploaded folder
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
                    embeddings = None
                    file_type = None
                    if file.suffix == '.txt':  # For text files
                        file_content, embeddings = process_text_file(file)
                        file_type = 'text'
                    elif file.suffix == '.pdf':  # For PDF files
                        file_content, embeddings = process_pdf_file(file)
                        file_type = 'pdf'
                    elif file.suffix == '.docx':  # For DOCX files
                        file_content, embeddings = process_docx_file(file)
                        file_type = 'docx'

                    if file_content and embeddings:
                        store_in_pinecone(file.stem, file_content, embeddings, file_type)
                    else:
                        st.warning(f"No text found in {file.name}. Skipping.")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")

        st.success("Folder contents stored in Pinecone.")

if __name__ == "__main__":
    main()
