import json
import os
import shutil
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
import torch
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import uuid
import re

# Load secrets from Streamlit
api_key = st.secrets["pinecone"]["api_key"]
env = st.secrets["pinecone"]["ENV"]
index_name = st.secrets["pinecone"]["INDEX_NAME"]
hf_token = st.secrets["huggingface"]["token"]
storage_folder = "file_storage"  # Folder for external storage

# Initialize Pinecone instance
pc = Pinecone(api_key=api_key, environment=env)

# Initialize transformer model
model_name = "distilbert-base-uncased"  # A smaller, easier model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModel.from_pretrained(model_name, use_auth_token=hf_token)

# Make sure storage folder exists
os.makedirs(storage_folder, exist_ok=True)

def extract_project_details(content):
    """Extract project details like Project Title, Location, Budget, etc."""
    project_details = {
        "Project Title": None,
        "Project Location": None,
        "Project Duration": None,
        "Name of Work": None,
        "Location of Work": None,
        "Project Budget": None,
        "Project Description": None
    }
    
    project_details["Project Title"] = re.search(r"Project Title[:\*]?\s*(.*)", content)
    project_details["Project Location"] = re.search(r"Project Location[:\*]?\s*(.*)", content)
    project_details["Project Duration"] = re.search(r"Project Duration[:\*]?\s*(.*)", content)
    project_details["Name of Work"] = re.search(r"Name of Work[:\*]?\s*(.*)", content)
    project_details["Location of Work"] = re.search(r"Location of work[:\*]?\s*(.*)", content)
    project_details["Project Budget"] = re.search(r"Project Budget[:\*]?\s*(.*)", content)
    project_details["Project Description"] = re.search(r"Project Description[:\*]?\s*(.*)", content)
    
    for key in project_details:
        if project_details[key]:
            project_details[key] = project_details[key].group(1).strip()
    
    return project_details

def process_text_file(file_path):
    """Process text files and extract embeddings"""
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()

        project_details = extract_project_details(content)
        inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

        return project_details, embeddings
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

        project_details = extract_project_details(content)
        
        if content.strip():  # If there's any text content in the PDF
            inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
            return project_details, embeddings
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

        project_details = extract_project_details(content)

        if content.strip():  # If there's any text content in the DOCX
            inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
            return project_details, embeddings
        else:
            return None, None
    except Exception as e:
        st.error(f"Error processing DOCX file {file_path}: {e}")
        return None, None

def store_in_pinecone(file_name, project_details, embeddings):
    """Store the embeddings and project details in Pinecone"""
    try:
        if project_details and embeddings:
            vector = embeddings

            # Convert project_details dictionary to JSON string
            project_details_str = json.dumps(project_details)

            # Store structured project details as a JSON string and embeddings
            metadata = {
                "file_name": file_name,
                "file_type": "unknown",  # You can customize this if needed
                "project_details": project_details_str  # Store as a JSON string
            }

            if len(vector) == 768:
                index = pc.Index(index_name)
                index.upsert([(file_name, vector, metadata)])
                st.write(f"Stored {file_name} in Pinecone with project details.")
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
                    project_details = None
                    embeddings = None
                    if file.suffix == '.txt':  # For text files
                        project_details, embeddings = process_text_file(file)
                    elif file.suffix == '.pdf':  # For PDF files
                        project_details, embeddings = process_pdf_file(file)
                    elif file.suffix == '.docx':  # For DOCX files
                        project_details, embeddings = process_docx_file(file)

                    if project_details and embeddings:
                        store_in_pinecone(file.stem, project_details, embeddings)
                    else:
                        st.warning(f"No content found in {file.name}. Skipping.")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")

        st.success("Folder contents stored in Pinecone.")

if __name__ == "__main__":
    main()
