#!/opt/anaconda3/envs/LLM-PDF/bin/python

# 1. Start Ollama service : service start ollama
# 3. Run code : python adb_main.py file.pdf

# Import necessary libraries
from langchain.chat_models import LangChainDeprecationWarning
import warnings
warnings.simplefilter(action='ignore', category=LangChainDeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os, sys
import tempfile

from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOllama
from langchain.embeddings import FastEmbedEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata

input_file = sys.argv[1]

# Define the ChatPDF class
class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        # Initialize the ChatOllama model
        self.model = ChatOllama(model="mistral",base_url="http://127.0.0.1:11434")
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        # Define the prompt template for conversation
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
             maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        # Load PDF documents
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        # Split the documents into smaller chunks
        chunks = self.text_splitter.split_documents(docs)
        # Filter out complex metadata
        chunks = filter_complex_metadata(chunks)

        # create the open-source embedding function
        #embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")    

        # Create a vector store from the chunks with FastEmbed embeddings
        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        # Convert the vector store into a retriever with specified search parameters
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        # Define the conversation chain using the retriever and prompt
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                    | self.prompt
                    | self.model
                    | StrOutputParser())
            
    def ask(self, query: str):
    # Check if the conversation chain has been initialized
        if not self.chain:
            return "Please, add a PDF document first."

        # Invoke the conversation chain with the user query
        return self.chain.invoke(query)
    
    def clear(self):
    # Clear the vector store, retriever, and conversation chain
        self.vector_store = None
        self.retriever = None
        self.chain = None

# Initialize ChatPDF instance
chat_pdf = ChatPDF()

chat_pdf.ingest(input_file)
print("PDF successfully ingested!")

# Function to handle user queries
def answer_query(query):
    if not chat_pdf.chain:
        print("Please, add a PDF document first.")
        return
    answer = chat_pdf.ask(query)
    print(f"Answer: {answer}")

while(1):
    user_query = input("Enter your question : ")
    if user_query == "Clear":
        chat_pdf.clear()
    else:
        answer_query(user_query)

