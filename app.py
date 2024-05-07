import os
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import Pinecone as PC
from pinecone import Pinecone, ServerlessSpec
import pinecone
from langchain_community.llms import CTransformers,HuggingFaceEndpoint
from langchain.chains import  RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from src.prompt import *

app = Flask(__name__)

load_dotenv()


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

embeddings = download_hugging_face_embeddings()
Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"
docsearch = PC.from_existing_index(index_name,embeddings)

retriever = docsearch.as_retriever(search_kwargs = {"k":2})
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=['contest','question'],

)

chain_type_kwargs={"prompt":PROMPT}

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=512, temperature=0.8, token=HUGGINGFACEHUB_API_TOKEN
)

#create a chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, # for refinement
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    # chain_type_kwargs=chain_type_kwargs
)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa_chain({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(debug=True)