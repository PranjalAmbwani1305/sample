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
    """Extract important data such as keywords or key terms."""
    important_data = {}

    # Define a list of keywords or patterns to search for
    keywords = [
        r"(?:Budget|Cost|Amount|Total)\s*[:\*]?\s*(\d+[,.]?\d*)",  # Extract budget or cost
        r"(?:Start Date|Start)\s*[:\*]?\s*([\d{2}/\d{2}/\d{4}])",  # Extract start date
        r"(?:End Date|Completion Date|Finish)\s*[:\*]?\s*([\d{2}/\d{2}/\d{4}])",  # Extract end date
        r"(?:Contractor|Company|Vendor)\s*[:\*]?\s*(.*)",  # Extract contractor or company name
        r"(?:Location|Site)\s*[:\*]?\s*(.*)",  # Extract location
        r"(?:Scope of Work|Project Scope|Work Description)\s*[:\*]?\s*(.*)",  # Extract scope of work
        r"(?:Specifications|Technical Details|Requirements)\s*[:\*]?\s*(.*)",  # Extract technical details
        r"(?:Contact Person|Project Manager|Point of Contact)\s*[:\*]?\s*(.*)",  # Extract contact person
    ]

    # Search for each keyword or pattern in the content
    for keyword in keywords:
        match = re.search(keyword, content, re.IGNORECASE)
        if match:
            important_data[keyword] = match.group(1).strip()

    return important_data

def generate_embeddings(text):
    """Generate embeddings for the important data text using Hugging Face model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Use mean of token embeddings
    return embeddings.squeeze().numpy()  # Convert to numpy array for Pinecone

def process_text_file(file_path):
    """Process text files and extract important data."""
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()

        # If content is empty, return empty data and skip
        if not content.strip():
            return None

        important_data = extract_important_data(content)
        return important_data
    except Exception as e:
        st.error(f"Error processing text file {file_path}: {e}")
        return None

def process_pdf_file(file_path):
    """Extract text from PDF and generate important data."""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            content = ''
            for page in reader.pages:
                content += page.extract_text()

        # If content is empty, return empty data and skip
        if not content.strip():
            return None

        important_data = extract_important_data(content)
        return important_data
    except Exception as e:
        st.error(f"Error processing PDF file {file_path}: {e}")
        return None

def process_docx_file(file_path):
    """Extract text from DOCX and generate important data."""
    try:
        doc = Document(file_path)
        content = ''
        for para in doc.paragraphs:
            content += para.text

        # If content is empty, return empty data and skip
        if not content.strip():
            return None

        important_data = extract_important_data(content)
        return important_data
    except Exception as e:
        st.error(f"Error processing DOCX file {file_path}: {e}")
        return None

def store_in_pinecone(file_name, important_data):
    """Store important data in Pinecone."""
    try:
        if important_data:
            # Convert important data dictionary to JSON string
            important_data_str = json.dumps(important_data)

            # Generate embeddings for important data
            embeddings = generate_embeddings(important_data_str)

            metadata = {
                "file_name": file_name,
                "file_type": "unknown",  # You can customize this if needed
                "important_data": important_data_str  # Store as a JSON string
            }

            # Store metadata and embeddings (required by Pinecone)
            index = pc.Index(index_name)
            index.upsert([(file_name, embeddings.tolist(), metadata)])  # Upsert with correct vector

            st.write(f"Stored {file_name} in Pinecone with important data.")
        else:
            st.warning(f"No important data found for {file_name}. Skipping.")
    except Exception as e:
        st.error(f"Error storing {file_name} in Pinecone: {e}")

def main():
    st.title("Upload and Store Important Data in Pinecone")

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
                    if file.suffix == '.txt':  # For text files
                        important_data = process_text_file(file)
                    elif file.suffix == '.pdf':  # For PDF files
                        important_data = process_pdf_file(file)
                    elif file.suffix == '.docx':  # For DOCX files
                        important_data = process_docx_file(file)

                    if important_data:
                        store_in_pinecone(file.stem, important_data)
                    else:
                        st.warning(f"No important data found in {file.name}. Skipping.")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")

        st.success("Folder contents stored in Pinecone.")

if __name__ == "__main__":
    main()
