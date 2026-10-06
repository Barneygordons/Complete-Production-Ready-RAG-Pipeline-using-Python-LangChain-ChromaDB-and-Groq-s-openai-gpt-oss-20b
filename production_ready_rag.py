import os
from pathlib import Path
from urllib import response

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Ensure your API key is configured in your environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Directory where Chroma will store vectors on disk
PERSIST_DIRECTORY = "./chroma_db"
SOURCE_FILE = Path("random_text.txt")

# 1. Initialize Embeddings and the Vector Store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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

# 3. Create a Retriever

# Fetch the most relevant chunks from the file
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Helper function to format the retrieved documents into a single string
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# 4. Define the Prompt and the Model
prompt_template = """Use only the context below to answer the question.

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0, max_tokens=2048, api_key=GROQ_API_KEY)


# 5. Assemble LCEL Rag Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 6. Execute the query
query = "Summarize the content of the text file in one sentence based only on the context."
answer = rag_chain.invoke(query)

print("\nQuestion:", query)
print("Answer:", answer)
print(answer.content)