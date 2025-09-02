
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import ServerlessSpec
from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_data = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
text_chunk = text_split(minimal_docs)

embedding = download_embeddings()
index_name = "studybot"

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )
index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

# retriever = docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

# retrieved_docs = retriever.invoke("What is Line detection?")

# print(retrieved_docs)