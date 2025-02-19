import streamlit as st
import openai
import pinecone
import tiktoken
import uuid
import pdfplumber
from docx import Document

openai_api_key = st.secrets["openai"]["api_key"]
pinecone_api_key = st.secrets["pinecone"]["api_key"]
pinecone_env = st.secrets["pinecone"]["ENV"]
index_name = st.secrets["pinecone"]["INDEX_NAME"]

openai.api_key = openai_api_key


tokenizer = tiktoken.get_encoding("cl100k_base")


def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to chunk text
def chunk_text(text, max_tokens=500):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [" ".join(tokenizer.decode(chunk).split()) for chunk in chunks]

# Function to generate embeddings
def generate_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response["data"][0]["embedding"]

# Streamlit UI
st.title("File Upload & Embedding Storage in Pinecone")

uploaded_file = st.file_uploader("Upload a TXT, PDF, or DOCX file", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    st.write("Processing file...")

    # Read file content
    if uploaded_file.type == "text/plain":
        text = uploaded_file.getvalue().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    # Break into sections
    sections = chunk_text(text)

    # Generate embeddings and store in Pinecone
    for i, section in enumerate(sections):
        embedding = generate_embedding(section)
        doc_id = str(uuid.uuid4())  # Unique ID
        index.upsert([(doc_id, embedding, {"text": section})])

    st.success(f"Stored {len(sections)} sections in Pinecone!")

    # Search query
    query = st.text_input("Search for relevant sections:")
    if query:
        query_embedding = generate_embedding(query)
        results = index.query(query_embedding, top_k=5, include_metadata=True)
        
        st.subheader("Search Results:")
        for match in results["matches"]:
            st.write(match["metadata"]["text"])
