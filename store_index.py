from src.helper import load_pdf,text_split,download_hugging_face_embeddings
from langchain_pinecone import Pinecone as PC
from pinecone import Pinecone, ServerlessSpec
import pinecone
from dotenv import load_dotenv
import os



load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

# print(PINECONE_API_KEY)


index_name = "medical-chatbot"

extract_data = load_pdf("./data")
text_chunks = text_split(extract_data)
# embeddings = download_hugging_face_embeddings()
# Pinecone(api_key=PINECONE_API_KEY)
# docsearch = PC.from_existing_index(index_name,embeddings)