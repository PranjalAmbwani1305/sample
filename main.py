import os
import streamlit as st
import pinecone
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch

os.environ["PINECONE_API_KEY"] = st.secrets["pinecone"]["api_key"]

pinecone.init()

index_name = st.secrets["pinecone"]["INDEX_NAME"]
index = pinecone.Index(index_name)

model_name = "huggingface/Mixtral"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def process_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        content = file.read()

    inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

    return embeddings

def store_in_pinecone(file_name, file_content):
    vector = file_content
    index.upsert([(file_name, vector)])

def main():
    st.title("Folder Upload and Pinecone Storage")

    uploaded_folder = st.file_uploader("Choose a folder to upload", type=["zip"], accept_multiple_files=False)

    if uploaded_folder is not None:
        save_path = Path("local_upload_folder")
        
        if save_path.exists():
            shutil.rmtree(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / uploaded_folder.name, "wb") as f:
            f.write(uploaded_folder.getvalue())
        
        shutil.unpack_archive(save_path / uploaded_folder.name, save_path)

        st.write(f"Folder uploaded and unzipped to {save_path}. Processing files...")

        for file in Path(save_path).rglob('*.*'):
            if file.is_file():
                st.write(f"Processing {file.name}...")
                try:
                    file_content = process_file(file)
                    store_in_pinecone(file.stem, file_content)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")

        st.success("Folder contents stored in Pinecone.")

if __name__ == "__main__":
    main()
