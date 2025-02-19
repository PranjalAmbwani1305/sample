import streamlit as st
import openai
import pinecone
import tiktoken
import uuid
import pdfplumber
from docx import Document
from tqdm import tqdm

# --- Initialize API Keys ---
openai_api_key = st.secrets["openai"]["api_key"]
pinecone_api_key = st.secrets["pinecone"]["api_key"]
pinecone_env = st.secrets["pinecone"]["ENV"]
index_name = st.secrets["pinecone"]["INDEX_NAME"]

# --- Initialize OpenAI & Pinecone ---
openai.api_key = openai_api_key
index = pinecone.GRPCIndex(index_name)

# --- Tokenizer for Chunking ---
tokenizer = tiktoken.get_encoding("cl100k_base")

# --- Function to Extract Text ---
def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(file):
    """Extract text from a DOCX file."""
    doc = Document(file)
    return "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])

def extract_text(file, file_type):
    """Extract text based on file type."""
    if file_type == "text/plain":
        return file.getvalue().decode("utf-8").strip()
    elif file_type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file)
    else:
        return None

# --- Function to Chunk Text ---
def chunk_text(text, max_tokens=500):
    """Split text into smaller chunks based on token limits."""
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [" ".join(tokenizer.decode(chunk).split()) for chunk in chunks]

# --- Function to Generate Embeddings ---
def generate_embedding(text):
    """Generate OpenAI embeddings for given text."""
    response = openai.embeddings.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding

# --- Streamlit UI ---
st.title("📂 AI-Powered Document Search with Pinecone")

uploaded_file = st.file_uploader("Upload a TXT, PDF, or DOCX file", type=["txt", "pdf", "docx"])

if uploaded_file:
    st.write("🔄 **Processing file...**")

    # Extract text from file
    text = extract_text(uploaded_file, uploaded_file.type)
    if not text:
        st.error("⚠️ Unable to extract text from the file. Please upload a valid document.")
        st.stop()

    # Split into chunks
    sections = chunk_text(text)

    # Show progress bar
    progress_bar = st.progress(0)
    
    # Store embeddings in Pinecone
    for i, section in enumerate(tqdm(sections)):
        embedding = generate_embedding(section)
        doc_id = str(uuid.uuid4())  # Unique document ID
        index.upsert([(doc_id, embedding, {"text": section})])
        progress_bar.progress((i + 1) / len(sections))

    st.success(f"✅ Successfully stored {len(sections)} sections in Pinecone!")

    # --- Search Functionality ---
    query = st.text_input("🔍 Search for relevant content:")
    if query:
        query_embedding = generate_embedding(query)
        results = index.query(query_embedding, top_k=5, include_metadata=True)

        st.subheader("📜 **Search Results:**")
        if results["matches"]:
            for match in results["matches"]:
                st.write(f"🔹 {match['metadata']['text']}")
        else:
            st.warning("⚠️ No relevant results found.")
