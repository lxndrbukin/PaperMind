# PaperMind — AI Document Q&A Engine

An AI-powered **document question-answering application** built with Python and Streamlit. Upload any PDF and ask natural language questions about its contents. Powered by **Retrieval-Augmented Generation (RAG)** — responses are grounded in your document, not general knowledge.

🔗 **Live Demo**: [papermind-analyzer.streamlit.app](https://papermind-analyzer.streamlit.app/)

---

## Features

- **PDF Upload & Parsing**: Upload any PDF and extract its full text content
- **Intelligent Chunking**: Documents are split into overlapping chunks for precise retrieval
- **Vector Search**: Chunks are embedded and searched using cosine similarity via OpenAI's embedding model
- **Grounded Answers**: Responses are generated strictly from document context — no hallucinations
- **Chat History**: Persistent conversation history within a session with a clean chat interface
- **Multilingual Support**: Ask questions in any language about documents written in any language

---

## Tech Stack

**Backend / AI**
- Python 3.11+
- OpenAI API (`text-embedding-3-small` + `gpt-4o-mini`)
- PyMuPDF (`fitz`) — PDF parsing
- LangChain — recursive text chunking
- NumPy — cosine similarity computation

**Frontend**
- Streamlit

**Deployment**
- Streamlit Community Cloud

---

## How It Works

PaperMind uses a RAG (Retrieval-Augmented Generation) pipeline:

1. **Parse** — PDF is converted to raw text using PyMuPDF
2. **Chunk** — Text is split into overlapping 500-word chunks using LangChain's `RecursiveCharacterTextSplitter`
3. **Embed** — Each chunk is converted to a 1,536-dimension vector using OpenAI's `text-embedding-3-small`
4. **Store** — Chunks and their embeddings are stored in memory
5. **Retrieve** — User question is embedded and compared against stored chunks using cosine similarity
6. **Generate** — Top 3 most relevant chunks are passed to `gpt-4o-mini` as context to generate a grounded answer

---

## Requirements

- Python 3.11+
- OpenAI API key

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/lxndrbukin/papermind.git
cd papermind
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

```env
OPENAI_API_KEY=your_openai_api_key
```

---

## Usage

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## Project Structure

```
papermind/
│
├── app.py              # Full application — RAG pipeline + Streamlit UI
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (not committed)
└── README.md
```

---

## Future Improvements

- [ ] PostgreSQL + pgvector — persistent vector storage
- [ ] FastAPI backend — production-ready API layer
- [ ] React frontend — full custom UI
- [ ] Multi-document support — upload and query across multiple PDFs
- [ ] Source citations — display which page each answer came from
- [ ] User authentication — per-user document storage

---

## License

This project is open source and available for personal and educational use.
