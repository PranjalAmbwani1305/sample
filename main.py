import os
import json
import re
import torch
import pinecone
import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
from docx import Document

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
pinecone.init(api_key=api_key, environment="us-west1-gcp")  # Adjust environment as needed
index_name = "tender-index"
pc = Pinecone(api_key=api_key)

# Load tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def extract_project_details(content):
    """Extract only the most important project details from content."""
    project_details = {}

    # Use regex to capture important fields. If a field is found, store it in the dictionary.
    title_match = re.search(r"Project Title[:\*]?\s*(.*)", content)
    location_match = re.search(r"Project Location[:\*]?\s*(.*)", content)
    budget_match = re.search(r"Project Budget[:\*]?\s*(.*)", content)
    
    # Only add fields that contain actual values
    if title_match:
        project_details["Project Title"] = title_match.group(1).strip()
    if location_match:
        project_details["Project Location"] = location_match.group(1).strip()
    if budget_match:
        project_details["Project Budget"] = budget_match.group(1).strip()

    # Return only the fields that were populated
    return project_details

def store_in_pinecone(file_name, project_details, embeddings):
    """Store embeddings and project details in Pinecone."""
    try:
        if project_details and embeddings:
            vector = embeddings

            # Convert project details to JSON if not empty
            if project_details:
                project_details_str = json.dumps(project_details)
            else:
                project_details_str = "No relevant details"

            # Store metadata as JSON string and embeddings
            metadata = {
                "file_name": file_name,
                "project_details": project_details_str
            }

            # Ensure the embeddings are of the correct dimension (e.g., 768 for BERT)
            if len(vector) == 768:
                index = pc.Index(index_name)
                index.upsert([(file_name, vector, metadata)])
                st.write(f"Stored {file_name} in Pinecone with relevant project details.")
            else:
                st.error(f"Invalid vector dimension for {file_name}. Expected 768, got {len(vector)}.")
        else:
            st.warning(f"No meaningful content found for {file_name}. Skipping.")
    except Exception as e:
        st.error(f"Error storing {file_name} in Pinecone: {e}")

def process_pdf_file(file_path):
    """Extract text from PDF and generate embeddings."""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            content = ''
            for page in reader.pages:
                content += page.extract_text()

        # If content is empty, return empty details and skip
        if not content.strip():
            return None, None

        project_details = extract_project_details(content)

        inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

        return project_details, embeddings
    except Exception as e:
        st.error(f"Error processing PDF file {file_path}: {e}")
        return None, None

def process_docx_file(file_path):
    """Extract text from DOCX file."""
    try:
        document = Document(file_path)
        content = ''
        for para in document.paragraphs:
            content += para.text + '\n'

        # If content is empty, return empty details and skip
        if not content.strip():
            return None, None

        project_details = extract_project_details(content)

        inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

        return project_details, embeddings
    except Exception as e:
        st.error(f"Error processing DOCX file {file_path}: {e}")
        return None, None

def process_txt_file(file_path):
    """Extract text from TXT file."""
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        # If content is empty, return empty details and skip
        if not content.strip():
            return None, None

        project_details = extract_project_details(content)

        inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

        return project_details, embeddings
    except Exception as e:
        st.error(f"Error processing TXT file {file_path}: {e}")
        return None, None

def handle_file(file_path):
    """Determine file type and process accordingly."""
    file_extension = file_path.split('.')[-1].lower()

    if file_extension == 'pdf':
        return process_pdf_file(file_path)
    elif file_extension == 'docx':
        return process_docx_file(file_path)
    elif file_extension == 'txt':
        return process_txt_file(file_path)
    else:
        st.warning(f"Unsupported file type: {file_extension}. Skipping {file_path}.")
        return None, None

def main():
    """Main function to process uploaded files and store in Pinecone."""
    st.title("Tender Data Extraction and Storage")

    # Upload folder containing files (PDF, DOCX, TXT)
    uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save uploaded file to local storage
            file_path = os.path.join("uploaded_files", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process the file
            project_details, embeddings = handle_file(file_path)

            # Store in Pinecone if valid project details and embeddings are found
            if project_details and embeddings:
                store_in_pinecone(uploaded_file.name, project_details, embeddings)
            else:
                st.warning(f"Skipping {uploaded_file.name} as no meaningful content was found.")

if __name__ == "__main__":
    main()
