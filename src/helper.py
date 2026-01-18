from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vector_store(docs, embeddings):
    return FAISS.from_documents(docs, embeddings)


def save_vector_store(db, path="faiss_index"):
    db.save_local(path)


def load_vector_store(path, embeddings):
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
