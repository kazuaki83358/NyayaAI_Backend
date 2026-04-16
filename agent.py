import os
import shutil
import time
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

load_dotenv()

# -------------------------
# Configuration
# -------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATA_DIR = "./data"
DB_DIR = "./chroma_db"

# -------------------------
# Global Objects
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

# Use local embeddings to avoid API rate limits
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Shared memory for persistence within the session
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

vector_store = None
qa_chain = None

# Custom System Prompt for Legal Personality
template = """You are a highly specialized Indian Law AI Assistant. Your goal is to provide accurate legal information based ONLY on the provided context (BNS, BNSS, Sakshya, and the Constitution of India).

Guidelines:
1. Always act as a professional legal expert, not a general AI.
2. If the answer is in the context, cite the specific sections or articles.
3. If you don't know the answer from the context, state that you cannot find it in the current legal documents provided, but do not make up information.
4. Keep the tone professional, authoritative, and helpful.

Context: {context}
Chat History: {chat_history}
Question: {question}

Law Agent Answer:"""

QA_PROMPT = PromptTemplate(
    template=template, input_variables=["context", "chat_history", "question"]
)

# -------------------------
# RAG Helper Functions
# -------------------------
def initialize_vector_db(force_reindex=False):
    """
    Initialize the vector database. If force_reindex is True, delete existing DB and recreate.
    """
    global vector_store, qa_chain

    # Check if DB_DIR already exists and we are not forcing re-index
    if not force_reindex and os.path.exists(DB_DIR):
        print("Loading existing vector database...")
        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        # Clear existing DB if re-indexing
        if os.path.exists(DB_DIR):
            print("Removing old database for re-indexing...")
            # Close connection by deleting object if exists
            vector_store = None
            # Small delay to ensure file locks are released
            time.sleep(1)
            try:
                shutil.rmtree(DB_DIR)
            except Exception as e:
                print(f"Warning: Could not delete {DB_DIR}: {e}")

        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            print(f"Created {DATA_DIR} directory. Please add your legal PDF files there.")
            return False, "Data directory created. Please add PDF files."

        pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
        if not pdf_files:
            return False, f"No PDF files found in {DATA_DIR}. Please add some legal documents."

        print(f"Loading {len(pdf_files)} PDF files...")
        all_docs = []
        for pdf in pdf_files:
            try:
                loader = PyPDFLoader(os.path.join(DATA_DIR, pdf))
                all_docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {pdf}: {e}")

        if not all_docs:
            return False, "Could not extract any content from the PDF files."

        # Split text into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)

        print(f"Creating vector database with {len(splits)} chunks (using local embeddings)...")
        
        # Local embeddings are fast and free
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=DB_DIR
        )

    # Create the Retrieval Chain
    if vector_store:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )
        return True, "Agent initialized successfully!"
    
    return False, "Initialization failed."

# Removed auto-initialize to prevent file locks on Windows during re-indexing

def run_agent(user_input):
    """
    Main function to interact with the Indian Law Agent.
    """
    global qa_chain
    if not qa_chain:
        # Try initializing if not yet set up
        success, message = initialize_vector_db()
        if not success:
            return f"I'm not ready yet! {message}"

    try:
        response = qa_chain.invoke({"question": user_input})
        return response['answer']
    except Exception as e:
        return f"Error occurred: {str(e)}"
