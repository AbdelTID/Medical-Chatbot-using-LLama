from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings



#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    return documents


# Create text chunks
def text_split(extract_data):
    text_splitter  = RecursiveCharacterTextSplitter(chunk_size =500,chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extract_data)
    return text_chunks

# download embeddding model
def download_hugging_face_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # model_kwargs = {'device': 'gpu'}
    # encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        # model_kwargs=model_kwargs,
        # encode_kwargs=encode_kwargs
    )
    return hf