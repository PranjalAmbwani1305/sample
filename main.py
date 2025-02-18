import os
import shutil
from pathlib import Path
import pinecone
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
pinecone.init(api_key=api_key, environment=env)

# Ensure the index exists in Pinecone
index = pinecone.Index(index_name)  # Access the Pinecone index

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

        return embeddings
    except Exception as e:
        st.error(f"Error processing text file {file_path}: {e}")
        return None

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
            return embeddings
        else:
            return None
    except Exception as e:
        st.error(f"Error processing PDF file {file_path}: {e}")
        return None

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
            return embeddings
        else:
            return None
    except Exception as e:
        st.error(f"Error processing DOCX file {file_path}: {e}")
        return None

def store_in_pinecone(file_name, file_content):
    """Store the embeddings in Pinecone"""
    try:
        if file_content:
            vector = file_content
            # Ensure the vector is of the correct dimension (based on your model output)
            if len(vector) == 768:  # For distilbert-base-uncased, the output dimension is 768
                index.upsert([(file_name, vector)])  # Store the vector in Pinecone index
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
                    if file.suffix == '.txt':  # For text files
                        file_content = process_text_file(file)
                    elif file.suffix == '.pdf':  # For PDF files
                        file_content = process_pdf_file(file)
                    elif file.suffix == '.docx':  # For DOCX files
                        file_content = process_docx_file(file)

                    if file_content:
                        store_in_pinecone(file.stem, file_content)
                    else:
                        st.warning(f"No text found in {file.name}. Skipping.")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")

        st.success("Folder contents stored in Pinecone.")

if __name__ == "__main__":
    main()
