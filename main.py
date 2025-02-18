import os
import streamlit as st
import pinecone
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import fitz
import docx

api_key = st.secrets["pinecone"]["api_key"]
env = st.secrets["pinecone"]["ENV"]
index_name = st.secrets["pinecone"]["INDEX_NAME"]
hf_token = st.secrets["huggingface"]["token"]

pinecone.init(api_key=api_key, environment=env)
index = pinecone.Index(index_name)

model_name = "mistralai/Mixtral-8x7B"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModel.from_pretrained(model_name, use_auth_token=hf_token)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

def read_file(file_path):
    try:
        if file_path.suffix.lower() == ".txt":
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        elif file_path.suffix.lower() == ".pdf":
            doc = fitz.open(str(file_path))
            return "\n".join([page.get_text("text") for page in doc])
        elif file_path.suffix.lower() == ".docx":
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            st.warning(f"Unsupported file type: {file_path.suffix}")
            return None
    except Exception as e:
        st.error(f"Error reading {file_path.name}: {e}")
        return None

def process_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings

def store_in_pinecone(file_name, file_content):
    vector = file_content
    index.upsert([(file_name, vector)])

def main():
    st.title("Tender Bot - Upload & Process Documents")

    uploaded_folder = st.file_uploader("Upload a ZIP folder containing tenders", type=["zip"])

    if uploaded_folder is not None:
        save_path = Path("local_upload_folder")

        if save_path.exists():
            shutil.rmtree(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / uploaded_folder.name, "wb") as f:
            f.write(uploaded_folder.getvalue())

        try:
            shutil.unpack_archive(save_path / uploaded_folder.name, save_path)
        except Exception as e:
            st.error(f"Error extracting ZIP file: {e}")
            return

        st.write(f"Folder extracted to {save_path}. Processing files...")

        for file in Path(save_path).rglob("*.*"):
            if file.is_file():
                text_content = read_file(file)
                if text_content:
                    try:
                        st.write(f"Processing {file.name}...")
                        embeddings = process_text(text_content)
                        store_in_pinecone(file.stem, embeddings)
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {e}")

        st.success("All valid files processed and stored in Pinecone.")

if __name__ == "__main__":
    main()
