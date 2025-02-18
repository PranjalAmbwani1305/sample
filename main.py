import os
import streamlit as st
import pinecone
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch

# Initialize Pinecone using the API key stored in Streamlit secrets
api_key = st.secrets["pinecone"]["api_key"]

# Initialize Pinecone
pinecone.init(api_key=api_key)

# Index setup
index_name = "tender_data"
dimension = 348  # You can adjust this depending on the type of data you're storing

# Create the index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=dimension)

index = pinecone.Index(index_name)

# Load the tokenizer and model for vectorization
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def process_file(file_path):
    """Process file content into embeddings (vectorized format)."""
    with open(file_path, 'r') as file:
        content = file.read()

    # Tokenize and create embeddings
    inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

    return embeddings

def store_in_pinecone(file_name, file_content):
    """Store the file content as a vector in Pinecone."""
    vector = file_content  # Already a vector from the process_file function
    index.upsert([(file_name, vector)])

def main():
    st.title("Folder Upload and Pinecone Storage")

    uploaded_folder = st.file_uploader("Choose a folder to upload", type=["zip"], accept_multiple_files=False)

    if uploaded_folder is not None:
        # Define the local save path for the folder
        save_path = Path("local_upload_folder")
        
        # Clear previous folder if any exists
        if save_path.exists():
            shutil.rmtree(save_path)  # Remove the old folder
        save_path.mkdir(parents=True, exist_ok=True)

        # Write the uploaded zip file to the local directory
        with open(save_path / uploaded_folder.name, "wb") as f:
            f.write(uploaded_folder.getvalue())
        
        # Unzip the file in the specified local path
        shutil.unpack_archive(save_path / uploaded_folder.name, save_path)

        st.write(f"Folder uploaded and unzipped to {save_path}. Processing files...")

        # Process and store each file in Pinecone
        for file in Path(save_path).rglob('*.*'):
            st.write(f"Processing {file.name}...")
            file_content = process_file(file)
            store_in_pinecone(file.stem, file_content)

        st.success("Folder contents stored in Pinecone.")

if __name__ == "__main__":
    main()
