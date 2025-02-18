import os
import shutil
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
import torch
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import boto3

# Load secrets from Streamlit
api_key = st.secrets["pinecone"]["api_key"]
env = st.secrets["pinecone"]["ENV"]
index_name = st.secrets["pinecone"]["INDEX_NAME"]
hf_token = st.secrets["huggingface"]["token"]
aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
bucket_name = st.secrets["aws"]["bucket_name"]

# Initialize Pinecone instance
pc = Pinecone(api_key=api_key, environment=env)

model_name = "distilbert-base-uncased"  # A smaller, easier model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModel.from_pretrained(model_name, use_auth_token=hf_token)

# Initialize Pinecone index
index = pc.Index(index_name)

# Initialize AWS S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

def upload_to_s3(content, file_name):
    """Upload full content to AWS S3 and return the URL"""
    try:
        # Store the file content in S3
        s3_client.put_object(Body=content, Bucket=bucket_name, Key=file_name)
        return f"https://{bucket_name}.s3.amazonaws.com/{file_name}"
    except Exception as e:
        st.error(f"Error uploading to S3: {e}")
        return None

def process_pdf_file(file_path):
    """Extract text from PDF and generate embeddings with preview and URL in metadata"""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            content = ''
            for page in reader.pages:
                content += page.extract_text()

        if content.strip():  # If there's text content in the PDF
            # Generate embeddings
            inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

            # Store the content in external storage and get the URL
            file_url = upload_to_s3(content, file_path.name)

            # Metadata includes the preview and the URL to the full content
            metadata = {
                "file_name": file_path.name,
                "file_type": "pdf",
                "content_preview": content[:200],  # Only the preview of the content
                "file_url": file_url  # URL to the full content stored externally
            }

            return embeddings, metadata
        else:
            return None, None
    except Exception as e:
        st.error(f"Error processing PDF file {file_path}: {e}")
        return None, None

def store_in_pinecone(file_name, file_content, metadata):
    """Store the embeddings and metadata (with preview and file URL) in Pinecone"""
    try:
        if file_content:
            vector = file_content
            if len(vector) == 768:  # Ensure correct embedding dimension
                # Storing both vector and metadata (including preview and URL) in Pinecone
                index.upsert([(file_name, vector, metadata)])
                st.write(f"Stored {file_name} with preview and URL in Pinecone.")
            else:
                st.error(f"Invalid vector dimension for {file_name}. Expected 768, got {len(vector)}.")
        else:
            st.warning(f"No content to store for {file_name}. Skipping.")
    except Exception as e:
        st.error(f"Error storing {file_name} in Pinecone: {e}")

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
                    if file.suffix == '.pdf':  # For PDF files
                        file_content, metadata = process_pdf_file(file)

                    if file_content:
                        store_in_pinecone(file.stem, file_content, metadata)
                    else:
                        st.warning(f"No text found in {file.name}. Skipping.")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")

        st.success("Folder contents stored in Pinecone.")

if __name__ == "__main__":
    main()
