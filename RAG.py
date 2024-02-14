import os
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain import HuggingFaceHub
from langchain_openai import OpenAI
from transformers import AutoTokenizer
import requests
import os
from langchain.text_splitter import NLTKTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
import nltk

nltk.download('punkt')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = input("Please input a huggingface API key: ")
os.environ["OPENAI_API_KEY"] = userdata.get('Please input a OpenAI API key: ')

def setup_directory(directory='docs'):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Download PDF (Provided in the question)
def pdf_download(url, directory, file_name):
    response = requests.get(url)
    if response.status_code == 200:
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'wb') as f:
            f.write(response.content)

# Load documents from directory
def load_documents_from_directory(directory):
    loader = PyPDFDirectoryLoader(directory)
    return loader.load()

# Initialize embedding model and tokenizer
def initialize_embedding_model():
    embeddings_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': False},
    )
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    return embeddings_model, tokenizer

# Split texts using NLTK
def split_texts(docs):
    nltk_splitter = NLTKTextSplitter(chunk_size=800)
    return nltk_splitter.split_documents(docs)

# Create and save FAISS vector store
def create_and_save_faiss_index(texts, embedding_model):
    db = FAISS.from_documents(texts, embedding_model)
    db.save_local("faiss_index")

# Turn DB into a retriever
def create_retriever():
    db = FAISS.load_local("faiss_index", bge_large_embeddings)  # Assuming bge_large_embeddings is globally accessible
    return db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5, "k": 3})

# Create the QA chain
def create_qa_chain(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True)

# Execute QA query
def execute_query(qa_chain, query):
    return qa_chain({"query": query})

def main():
    llm = OpenAI(api_key = os.environ["OPENAI_API_KEY"],)
    setup_directory(directory='docs')
    
    url = 'https://www.ama-assn.org/system/files/ama-future-health-report.pdf'
    url2 = 'https://www.cdc.gov/nchs/data/factsheets/factsheet_hiac.pdf'
    pdf_download(url, 'docs', "AMA_report.pdf")
    pdf_download(url2, 'docs', "CDC_facts.pdf")
    
    docs = load_documents_from_directory('/content/docs')
    embedding_model, tokenizer = initialize_embedding_model()
    texts = split_texts(docs)
    create_and_save_faiss_index(texts, embedding_model)
    
    retriever = create_retriever()
    qa_chain = create_qa_chain(llm, retriever)
    query = "What are the differences in health insurance coverage rates among different racial and ethnic groups as per the first 6 months of 2016?"
    result = execute_query(qa_chain, query)
    
    print(result)

# Call main function to execute the process
if __name__ == "__main__":
    main()