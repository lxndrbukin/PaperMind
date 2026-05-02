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

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": "Answer the user's query only using the provided context"
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