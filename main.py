import streamlit as st
from pinecone import Pinecone
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone

# Load API keys from Streamlit secrets
PINECONE_API_KEY = st.secrets["pinecone"]["api_key"]
PINECONE_ENV = st.secrets["pinecone"]["env"]
PINECONE_INDEX = st.secrets["pinecone"]["index_name"]
HF_TOKEN = st.secrets["huggingface"]["token"]

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Initialize Hugging Face embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF using LangChain's PyPDFLoader."""
    loader = PyPDFLoader(uploaded_file)
    docs = loader.load()
    return docs

def process_and_store_pdf(uploaded_file):
    """Extract text, generate embeddings, and store in Pinecone."""
    docs = extract_text_from_pdf(uploaded_file)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Store embeddings in Pinecone
    vectorstore = LangchainPinecone.from_documents(chunks, embedding_model, index_name=PINECONE_INDEX)

    return len(chunks)

# Streamlit UI
st.title("📄 PDF to Pinecone Storage")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    num_chunks = process_and_store_pdf(uploaded_file)
    st.success(f"Stored {num_chunks} chunks in Pinecone! 🚀")
