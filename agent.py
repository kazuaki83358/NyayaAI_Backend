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

# Metadata mapping for different PDF documents
pdf_metadata = {
    "the_constitution_of_india.pdf": {
        "category": "constitution",
        "law_name": "Constitution of India"
    },
    "bns.pdf": {
        "category": "criminal_law",
        "law_name": "Bharatiya Nyaya Sanhita"
    },
    "bnss.pdf": {
        "category": "procedure_law",
        "law_name": "Bharatiya Nagarik Suraksha Sanhita"
    },
    "sakshya.pdf": {
        "category": "evidence_law",
        "law_name": "Bharatiya Sakshya Adhiniyam"
    },
    "it_act_2000_updated.pdf": {
        "category": "cybercrime",
        "law_name": "Information Technology Act"
    },
    "Consumer_Protection_Act.pdf": {
        "category": "consumer_law",
        "law_name": "Consumer Protection Act"
    },
    "payment_of_wages_act_1936.pdf": {
        "category": "labour_law",
        "law_name": "Payment of Wages Act"
    },
    "Motor_Vehicles_Act.pdf": {
        "category": "traffic_law",
        "law_name": "Motor Vehicles Act"
    }
}

# -------------------------
# Global Objects
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

# Use a multilingual embedding model to better support Hindi and Hinglish queries
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

# Shared memory for persistence within the session
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

vector_store = None
qa_chain = None

# Custom System Prompt for Legal Personality
template = """You are Nyaya AI, a highly specialized Indian Law Assistant. Your goal is to provide accurate legal information based ONLY on the provided context (Constitution of India, BNS, BNSS, Sakshya, IT Act, Consumer Protection Act, Motor Vehicles Act, etc.).

When asked about "rights" or "fundamental rights", prioritize information from Part III of the Constitution (Articles 12-35).

Tone and Style:
1. For simple greetings (e.g., "Hi", "Hello", "Hey", "Greetings"), respond with a professional and welcoming message as Nyaya AI. Introduce yourself as a legal assistant and ask how you can help with Indian law today. Do NOT use the legal breakdown format for these.
2. For actual legal inquiries or situational queries, start with a professional and empathetic legal greeting (e.g., "Greetings. Based on the legal documents, here is the information regarding your situation...").
3. If the user describes a situation, identify the likely legal offences and sections immediately.
4. If the user asks in Hindi or Hinglish, respond in the same language.
5. If the question is about Constitutional Rights, use this format:
   - Relevant Article(s): [List the articles]
   - Meaning: [Explain the right clearly]
   - Remedies: [Mention Article 32 or 226 if applicable]
   - Disclaimer: [Short legal disclaimer]

6. For situational or criminal queries, use this STRICT format:
   - Likely Offence: [Name of the offence]
   - Relevant Section(s): [Section numbers and Act name]
   - Punishment: [Briefly mention the penalty]
   - Recommended Next Steps: [Actionable advice]
   - Disclaimer: [Short legal disclaimer]

7. Keep the total answer under 250 words.
8. If the answer is not in the context, state that you cannot find it in the specific Acts provided, but do not hallucinate.

Context: {context}
Chat History: {chat_history}
Question: {question}

Nyaya AI Answer:"""

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
    global vector_store, qa_chain, DB_DIR

    # Check if DB_DIR already exists and we are not forcing re-index
    if not force_reindex and os.path.exists(DB_DIR):
        print("Loading existing vector database...")
        try:
            vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        except Exception as e:
            print(f"Error loading existing database: {e}. Falling back to re-indexing.")
            force_reindex = True

    if force_reindex or not os.path.exists(DB_DIR):
        # Clear existing DB if re-indexing
        if os.path.exists(DB_DIR):
            print("Removing old database for re-indexing...")
            
            # Explicitly close the client if possible
            if vector_store is not None:
                try:
                    # Access the underlying client to close it
                    if hasattr(vector_store, "_client"):
                        vector_store._client.close()
                    elif hasattr(vector_store, "_client_settings"):
                        # Older versions or different internal structure
                        pass
                except Exception as e:
                    print(f"Error closing chroma client: {e}")

            vector_store = None
            qa_chain = None
            
            # Force garbage collection to help release file locks on Windows
            import gc
            gc.collect()
            
            # Small delay to ensure file locks are released
            time.sleep(2)
            
            try:
                shutil.rmtree(DB_DIR)
                print("Old database deleted successfully.")
            except Exception as e:
                print(f"Warning: Could not delete {DB_DIR}: {e}")
                # If we can't delete it, we MUST use a different directory to avoid corruption
                # because the current one is likely corrupted or locked.
                new_db_dir = f"{DB_DIR}_{int(time.time())}"
                print(f"Using fallback directory: {new_db_dir}")
                # We update the global DB_DIR so future loads use this new one
                # Note: In a real production app, you'd manage this path in a config file
                DB_DIR = new_db_dir

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
                docs = loader.load()
                
                # Get metadata from mapping or use default
                metadata = pdf_metadata.get(pdf, {
                    "category": "general",
                    "law_name": pdf
                })

                for doc in docs:
                    doc.metadata["category"] = metadata["category"]
                    doc.metadata["law_name"] = metadata["law_name"]
                    doc.metadata["source_file"] = pdf
                    
                    # Special tagging for the Constitution if category is constitution
                    if metadata["category"] == "constitution":
                        page_num = doc.metadata.get("page", 0)
                        if 24 <= page_num <= 32:
                            doc.metadata["category"] = "fundamental_rights"
                
                all_docs.extend(docs)
            except Exception as e:
                print(f"Error loading {pdf}: {e}")

        if not all_docs:
            return False, "Could not extract any content from the PDF files."

        # Split text into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)

        print(f"Creating vector database with {len(splits)} chunks (using local embeddings)...")
        
        try:
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=DB_DIR
            )
        except Exception as e:
            return False, f"Failed to create vector database: {e}"

    # Create the Retrieval Chain
    if vector_store:
        # Use a retriever with a bit more flexibility
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=base_retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )
        return True, "Agent initialized successfully!"
    
    return False, "Initialization failed."

# Removed auto-initialize to prevent file locks on Windows during re-indexing

def run_agent(user_input):
    """
    Main function to interact with the Indian Law Agent with dynamic retrieval logic.
    """
    global qa_chain, vector_store
    if not qa_chain:
        # Try initializing if not yet set up
        success, message = initialize_vector_db()
        if not success:
            return f"I'm not ready yet! {message}"

    try:
        query = user_input.lower()
        
        # User-specified category detection logic
        if any(word in query for word in ["upi", "hack", "instagram", "cyber", "online", "otp", "scam"]):
            filter_category = "cybercrime"
        elif any(word in query for word in ["consumer", "refund", "order", "delivery", "warranty", "seller"]):
            filter_category = "consumer_law"
        elif any(word in query for word in ["salary", "wages", "employer", "boss", "company", "job"]):
            filter_category = "labour_law"
        elif any(word in query for word in ["traffic", "challan", "driving", "license", "accident"]):
            filter_category = "traffic_law"
        elif any(word in query for word in ["right", "article", "constitution", "freedom", "equality"]):
            # Use fundamental_rights for specific rights-related keywords
            if any(word in query for word in ["right", "freedom", "equality"]):
                filter_category = "fundamental_rights"
            else:
                filter_category = "constitution"
        elif any(word in query for word in ["arrest", "police", "warrant", "fir", "bail", "arrested"]):
            filter_category = "procedure_law"
        elif any(word in query for word in ["beat", "attack", "murder", "fight", "threat", "knife"]):
            filter_category = "criminal_law"
        else:
            filter_category = None

        if filter_category:
            print(f"Applying filter for category: {filter_category}")
            qa_chain.retriever = vector_store.as_retriever(
                search_kwargs={"k": 5, "filter": {"category": filter_category}}
            )
        else:
            # Default retrieval for other legal queries
            qa_chain.retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        response = qa_chain.invoke({"question": user_input})
        return response['answer'], filter_category
    except Exception as e:
        return f"Error occurred: {str(e)}", None
