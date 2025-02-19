import streamlit as st
import pdfplumber
from pinecone import Pinecone
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
    """Extracts text from a PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def process_and_store_pdf(uploaded_file):
    """Extract text, generate embeddings, and store in Pinecone."""
    text = extract_text_from_pdf(uploaded_file)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    # Convert to LangChain document format
    docs = [{"page_content": chunk} for chunk in chunks]

    # Store embeddings in Pinecone
    vectorstore = LangchainPinecone.from_documents(docs, embedding_model, index_name=PINECONE_INDEX)

    return chunks  # Return chunks for display

# Streamlit UI
st.title("📄 PDF to Pinecone Storage")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    chunks = process_and_store_pdf(uploaded_file)
    num_chunks = len(chunks)
    
    st.success(f"✅ Stored {num_chunks} chunks in Pinecone! 🚀")

    # Display chunks in an expandable section
    with st.expander("📌 View Extracted Chunks"):
        for i, chunk in enumerate(chunks):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(chunk)
            st.markdown("---")  # Add a separator
