import os
import streamlit as st
import pinecone
from pinecone import Pinecone as PineconeClient
import shutil
from pathlib import Path

# Initialize Pinecone using the API key stored in Streamlit secrets
api_key = st.secrets["pinecone"]["api_key"]

# Initialize Pinecone
pinecone.init(api_key=api_key)

# Index setup
index_name = "tender_data"
dimension = 384  # You can adjust this depending on the type of data you're storing

# Create the index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=dimension)

index = pinecone.Index(index_name)

def process_file(file_path):
    """Dummy function to simulate processing of the file into a vector."""
    with open(file_path, 'r') as file:
        content = file.read()
    return content  # Replace with actual vectorization (e.g., embeddings)

def store_in_pinecone(file_name, file_content):
    """Store the file content as a vector in Pinecone."""
    vector = [ord(c) for c in file_content[:dimension]]  # Simple vectorization; replace with real embedding
    index.upsert([(file_name, vector)])

def main():
    st.title("Folder Upload and Pinecone Storage")

    uploaded_folder = st.file_uploader("Choose a folder to upload", type=["zip"], accept_multiple_files=False)

    if uploaded_folder is not None:
        # Clear previous folder if any exists
        folder_path = Path("uploaded_folder")
        if folder_path.exists():
            shutil.rmtree(folder_path)  # Remove the old folder
        folder_path.mkdir()

        # Write the uploaded zip file and unzip it
        with open(folder_path / uploaded_folder.name, "wb") as f:
            f.write(uploaded_folder.getvalue())
        shutil.unpack_archive(folder_path / uploaded_folder.name, folder_path)

        st.write("Folder uploaded and unzipped. Processing files...")

        # Process and store each file in Pinecone
        for file in Path(folder_path).rglob('*.*'):
            file_content = process_file(file)
            store_in_pinecone(file.stem, file_content)

        st.success("Folder contents stored in Pinecone.")

if __name__ == "__main__":
    main()
