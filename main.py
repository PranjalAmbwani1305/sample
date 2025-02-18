import os
import shutil
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
import torch
import streamlit as st

# Load secrets from Streamlit
api_key = st.secrets["pinecone"]["api_key"]
env = st.secrets["pinecone"]["ENV"]
index_name = st.secrets["pinecone"]["INDEX_NAME"]
hf_token = st.secrets["huggingface"]["token"]

# Initialize Pinecone instance
pc = Pinecone(api_key=api_key, environment=env)

# Initialize the model with Hugging Face, using a simpler model
model_name = "distilbert-base-uncased"  # A smaller, easier model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModel.from_pretrained(model_name, use_auth_token=hf_token)

def process_file(file_path):
    """Process the file and extract embeddings"""
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()

        inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

        return embeddings
    except Exception as e:
        st.error(f"Error processing file {file_path}: {e}")
        return None

def store_in_pinecone(file_name, file_content):
    """Store the embeddings in Pinecone"""
    try:
        vector = file_content
        index.upsert([(file_name, vector)])  # Store the vector in Pinecone index
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
                    file_content = process_file(file)
                    if file_content:
                        store_in_pinecone(file.stem, file_content)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")

        st.success("Folder contents stored in Pinecone.")

if __name__ == "__main__":
    main()
