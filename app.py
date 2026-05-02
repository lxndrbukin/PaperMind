import fitz
import os
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

INSTRUCTIONS = """
You are PaperMind, an intelligent document assistant. Your sole purpose is to answer questions based strictly on the content of the document provided to you as context.

CORE RULES:
- Answer ONLY using information found in the provided context. Never use your general knowledge to answer questions about the document.
- If the answer is not found in the context, say clearly: "I couldn't find that information in the document." Do not guess or infer beyond what is explicitly stated.
- Never fabricate facts, figures, names, dates, or any other information not present in the context.

RESPONSE STYLE:
- Be concise and direct. Answer the question asked without unnecessary padding.
- Use clear, professional language.
- When quoting or referencing specific parts of the document, indicate where the information comes from if possible.
- Format responses using markdown where it improves readability — use bullet points for lists, bold for key terms, and code blocks for technical content.

CONVERSATION:
- Remember the conversation history and use it to provide coherent, contextually aware follow-up answers.
- If a follow-up question refers to something mentioned earlier in the conversation, use that context naturally.
- If a question is ambiguous, ask for clarification before attempting an answer.

LANGUAGE:
- Always respond in the same language the user is writing in, regardless of the language the document is written in.
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": INSTRUCTIONS
        }
    ]

if "display_messages" not in st.session_state:
    st.session_state["display_messages"] = []

def parse_pdf(file):
	text = ""
	doc = fitz.open(stream=file.read(), filetype="pdf")
	for page in doc:
		text += page.get_text() + "\n"
	doc.close()
	return text

def chunk_text(text):
	splitter = RecursiveCharacterTextSplitter(
					chunk_size=500, 
					chunk_overlap=50
				)
	return splitter.split_text(text)

def embed_chunks(chunks):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    return [item.embedding for item in response.data]

def store(chunks, embeddings):
	result = []
	for idx, chunk in enumerate(chunks):
		result.append({
			"chunk": chunk,
			"embedding": embeddings[idx]
		})
	return result

def retrieve(query, stored_chunks):
	scores = []
	response = client.embeddings.create(
			model="text-embedding-3-small",
			input=[query]
	)
	query_embedding = response.data[0].embedding
	for chunk in stored_chunks:
		similarity = np.dot(query_embedding, chunk["embedding"]) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk["embedding"]))
		scores.append({
			"chunk": chunk,
			"score": similarity
		})
	return sorted(scores, key=lambda x: x["score"], reverse=True)[:3]

def generate(query, retrieved_chunks):
	text = ""
	for retrieved_chunk in retrieved_chunks:
		text += retrieved_chunk["chunk"]["chunk"]

	st.session_state["messages"].append({
		"role": "user",
		"content": f"Context: {text}\nUser query: {query}"
	})
	response = client.chat.completions.create(
		model="gpt-4o-mini",
		messages=st.session_state["messages"]
	)
	
	answer = response.choices[0].message.content
		
	st.session_state["messages"].append({
        "role": "assistant",
        "content": answer
    })
	
	st.session_state["display_messages"].append({"role": "user", "content": query})
	st.session_state["display_messages"].append({"role": "assistant", "content": answer})

	return answer

st.set_page_config(
    page_title="PaperMind",
    page_icon="🧠",
    layout="centered"
)

st.title("PaperMind")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
	with st.spinner("Processing PDF..."):
		text = parse_pdf(uploaded_file)
		chunks = chunk_text(text)
		embeddings = embed_chunks(chunks)
		st.session_state["stored_chunks"] = store(chunks, embeddings)
	st.write("File uploaded")

question = st.chat_input("Ask a question")

if question:
	if "stored_chunks" not in st.session_state:
		st.warning("Please upload a PDF first.")
	else:
		with st.spinner("Thinking..."):
			retrieved = retrieve(question, st.session_state["stored_chunks"])
			generate(question, retrieved)

for msg in st.session_state.get("display_messages", []):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])