import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from lanchain_openai import ChatOpenAI, OpenAIEmbeddings
from lanchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Ensure your API key is configured in your environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Directory where Chroma will store vectors on disk
PERSIST_DIRECTORY = "./chroma_db"
SOURCE_FILE = Path("random_text.txt")

#1. Initialize Embeddings and the Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize Chroma with a local storage path
vector_store = Chroma(
    collection_name="random_text_chunks",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIRECTORY, 
)

# 2. Reading the text file, splitting into chunks and finally embedding
if vector_store._collection.count() == 0:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Text file not found: {SOURCE_FILE}")

    file_text = SOURCE_FILE.read_text(encoding="utf-8")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(file_text)

    docs = [
        Document(page_content=chunk, metadata={"source": str(SOURCE_FILE)})
        for chunk in chunks
        if chunk.strip()
    ]

    if docs:
        vector_store.add_documents(docs)
        print(f"Loaded {len(docs)} text chunks from {SOURCE_FILE} and embedded them.")
    else:
        print(f"No text found in {SOURCE_FILE}.")
else:
    print(f"Loaded existing database with {vector_store._collection.count()} items.")

    