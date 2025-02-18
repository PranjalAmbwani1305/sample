import json
import os
import shutil
from pathlib import Path
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
import torch
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import uuid
import re

# Initialize Pinecone instance
api_key = st.secrets["pinecone"]["api_key"]
env = st.secrets["pinecone"]["ENV"]
index_name = st.secrets["pinecone"]["INDEX_NAME"]
hf_token = st.secrets["huggingface"]["token"]
storage_folder = "file_storage"  # Folder for external storage

pc = Pinecone(api_key=api_key, environment=env)

# Initialize Huggingface transformer model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModel.from_pretrained(model_name, use_auth_token=hf_token)

# Make sure storage folder exists
os.makedirs(storage_folder, exist_ok=True)

def extract_important_data(content):
    """Extract important data from tender content using regex."""
    important_data = {
        "Tender Title": None,
        "Tender Number": None,
        "Tender Description": None,
        "Tender Date": None
    }

    # Use regular expressions to find fields like "Tender Title", "Tender Number", etc.
    important_data["Tender Title"] = re.search(r"Tender Title[:\*]?\s*(.*)", content)
    important_data["Tender Number"] = re.search(r"Tender Number[:\*]?\s*(.*)", content)
    important_data["Tender Description"] = re.search(r"Tender Description[:\*]?\s*(.*)", content)
    important_data["Tender Date"] = re.search(r"Tender Date[:\*]?\s*(.*)", content)

    # Extract matched values from regex
    for key in important_data:
        if important_data[key]:
            important_data[key] = important_data[key].group(1).strip()

    # Return important data with None replaced with 'Not Available'
    return {key: (value if value else "Not Available") for key, value in important_data.items()}

def process_text_file(file_path):
    """Process text files and extract embeddings"""
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()

        # If content is empty, return empty data and skip
        if not content.strip():
            return None, None

        important_data = extract_important_data(content)
        inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

        return important_data, embeddings
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

        # If content is empty, return empty data and skip
        if not content.strip():
            return None, None

        important_data = extract_important_data(content)

        inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

        return important_data, embeddings
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

        # If content is empty, return empty data and skip
        if not content.strip():
            return None, None

        important_data = extract_important_data(content)

        inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

        return important_data, embeddings
    except Exception as e:
        st.error(f"Error processing DOCX file {file_path}: {e}")
        return None, None

def store_in_pinecone(file_name, important_data, embeddings):
    """Store embeddings and full content reference in Pinecone"""
    try:
        if important_data and embeddings:
            # Store full content in local storage
            content = important_data.get("Tender Description", "")
            
            # For simplicity, store the full content locally in "content_storage" folder
            content_storage_folder = "content_storage"
            os.makedirs(content_storage_folder, exist_ok=True)

            # Generate a unique filename for the content file
            content_file_path = os.path.join(content_storage_folder, f"{file_name}.txt")

            # Save the content to a text file
            with open(content_file_path, 'w', encoding="utf-8") as content_file:
                content_file.write(content)

            # Store a reference to the file (file path) in Pinecone
            content_reference = content_file_path  # Local file path

            # Create metadata with the reference
            metadata = {
                "file_name": file_name,
                "content_reference": content_reference,  # Reference to the content file
                "tender_title": important_data.get("Tender Title", "Not Available"),
                "tender_number": important_data.get("Tender Number", "Not Available")
            }

            vector = embeddings  # Directly using the list format embeddings

            # Ensure the embeddings are of the correct dimension (e.g., 768 for BERT)
            if len(vector) == 768:
                index = pc.Index(index_name)
                index.upsert([(file_name, vector, metadata)])
                st.write(f"Stored {file_name} in Pinecone with content reference.")
            else:
                st.error(f"Invalid vector dimension for {file_name}. Expected 768, got {len(vector)}.")
        else:
            st.warning(f"No content found for {file_name}. Skipping.")
    except Exception as e:
        st.error(f"Error storing {file_name} in Pinecone: {e}")

def main():
    st.title("Upload and Store Tender Data in Pinecone")

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
                    important_data = None
                    embeddings = None
                    if file.suffix == '.txt':  # For text files
                        important_data, embeddings = process_text_file(file)
                    elif file.suffix == '.pdf':  # For PDF files
                        important_data, embeddings = process_pdf_file(file)
                    elif file.suffix == '.docx':  # For DOCX files
                        important_data, embeddings = process_docx_file(file)

                    if important_data and embeddings:
                        store_in_pinecone(file.stem, important_data, embeddings)
                    else:
                        st.warning(f"No content found in {file.name}. Skipping.")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")

        st.success("Folder contents stored in Pinecone.")

if __name__ == "__main__":
    main()
