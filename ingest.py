import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PDF_FOLDER = "F1_regulations"

print("Loading all PDFs from F1_regulations folder...")
all_documents = []

for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        filepath = os.path.join(PDF_FOLDER, filename)
        print(f"  Loading: {filename}")
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        all_documents.extend(docs)

print(f"\nTotal pages loaded: {len(all_documents)}")

print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(all_documents)
print(f"Created {len(chunks)} chunks")

print("Embedding and storing in ChromaDB (this may take a few minutes)...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("\nDone! Your F1 knowledge base is ready.")
