import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

class BioRetriever:
    """
    Handles document ingestion and similarity search for bioinformatics data.
    """
    def __init__(self, openai_api_key):
        # Initialize embeddings with OpenAI's model
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.persist_directory = "./db/chroma_db"
        self.vector_db = None

    def ingest_documents(self, file_path):
        """
        Loads a PDF, splits it into chunks, and stores it in the vector database.
        """
        if not os.path.exists(file_path):
            return "File not found. Please check the path."

        # 1. Load the PDF document
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # 2. Split text into manageable chunks for the LLM
        # Bio-terms are complex, so we use a chunk size of 1000 with some overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150
        )
        texts = text_splitter.split_documents(documents)
        
        # 3. Create and persist the vector database
        self.vector_db = Chroma.from_documents(
            documents=texts, 
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return f"Successfully ingested {len(texts)} chunks from {file_path}."

    def search_relevant_context(self, query, k=3):
        """
        Retrieves the top k most relevant document snippets for a given query.
        """
        if self.vector_db is None:
            # Load existing DB if not already in memory
            self.vector_db = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings
            )
            
        docs = self.vector_db.similarity_search(query, k=k)
        return docs
