# Indian Law AI Agent ⚖️🇮🇳

An AI-powered legal assistant built with **LangChain**, **Google Gemini**, and **ChromaDB**. This agent uses **Retrieval-Augmented Generation (RAG)** to provide accurate legal information based on official Indian laws.

## 🚀 Features
- **RAG-based Question Answering**: Retrieves context from real legal PDFs.
- **Support for Multiple Acts**: Index Constitution, BNS, BNSS, BSA, IPC, CrPC, etc.
- **Conversational Memory**: Remembers the context of your legal consultation.
- **Streamlit UI**: Clean and intuitive chat interface.

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.8+
- [Google Gemini API Key](https://aistudio.google.com/app/apikey)

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Add Legal Data
Place all your legal PDF documents into the `data/` folder. Examples:
- `Constitution_of_India.pdf`
- `Bharatiya_Nyaya_Sanhita_2023.pdf`
- `IPC_1860.pdf`

### 4. Configuration
Create a `.env` file in the root directory and add your API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5. Run the Application
```bash
streamlit run app.py
```

### 6. Deploy on Render
This repo includes a [`render.yaml`](render.yaml) blueprint for Render.

1. Push the project to GitHub.
2. In Render, create a new `Web Service` from the repository.
3. Set the environment to use the blueprint, or copy these values if you deploy manually:
	- Build command: `pip install -r requirements.txt`
	- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
	- Python version: `3.11.11`
4. Add a secret environment variable named `GOOGLE_API_KEY` with your Gemini API key.
5. Deploy and wait for the first build to finish. The initial boot can take a while because the app loads the PDFs and builds the Chroma index.

For a smooth demo, use the paid starter plan or expect the free service to sleep between visits.

## 🏗️ Architecture
1. **Document Loading**: `PyPDFLoader` reads PDFs from the `data/` folder.
2. **Text Splitting**: `RecursiveCharacterTextSplitter` breaks documents into manageable chunks.
3. **Embeddings**: `GoogleGenerativeAIEmbeddings` converts text chunks into vector representations.
4. **Vector Store**: `Chroma` stores and retrieves relevant chunks based on user queries.
5. **LLM**: `Gemini 1.5 Flash` generates final responses using the retrieved legal context.

---
*Disclaimer: This AI is for educational purposes and project demonstration only. It is not a substitute for professional legal advice.*
