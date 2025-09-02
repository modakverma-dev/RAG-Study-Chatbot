from flask import Flask, render_template, jsonify, request
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from src.helper import download_embeddings
from src.prompt import system_prompt
from langchain_pinecone import PineconeVectorStore
# from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
# from src.prompt import *
import os

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embedding = download_embeddings()

index_name = "studybot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

retriever = docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})
# retrieved_docs = retriever.invoke("What is Line detection?")
# print(retrieved_docs)

generator = pipeline(
    "text-generation",
    model="google/flan-t5-base",   # flan-t5-large -> for better quality
    device=0                      
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}"),
    ]
)
chatModel = HuggingFacePipeline(pipeline=generator)
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input":msg})
    print("Response: ",response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)