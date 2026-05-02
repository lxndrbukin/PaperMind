import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv
import os

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