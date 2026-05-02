import fitz
import os
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

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
	response = client.chat.completions.create(
		model="gpt-4o-mini",
		messages=[
			{
				"role": "system",
				"content": "Answer the user's query only using the provided context"
			},
			{
				"role": "user",
				"content": f"Context: {text}\nUser query: {query}"
			}
		]
	)
	return response.choices[0].message.content