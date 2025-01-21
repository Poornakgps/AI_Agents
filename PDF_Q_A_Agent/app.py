from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from io import BytesIO

# FastAPI app
app = FastAPI()

class PDFQuestionAnswering:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API key is missing. Please provide a valid API key.")
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.vector_store = None
        self.chunks = None

    def extract_pdf_text(self, pdf_file):
        reader = PdfReader(pdf_file)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        if not text.strip():
            raise ValueError("No readable text found in the PDF.")
        return text

    def split_text(self, text):
        text_length = len(text)
        if text_length > 1e6:  # For large texts
            max_size = 1000
            chunk_overlap = 100
        else:
            max_size = 5000
            chunk_overlap = 500
        splitter = RecursiveCharacterTextSplitter(chunk_size=max_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)

    def build_vector_store(self, chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        return vector_store

    def load_qa_chain(self):
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3, google_api_key=self.api_key)
        chain = load_qa_chain(model, chain_type="map_reduce")
        return chain

    def ask_question(self, question, vector_store, chunks):
        docs = vector_store.similarity_search(question)
        context = "\n".join(chunks)
        chain = self.load_qa_chain()
        response = chain.invoke({"input_documents": docs, "context": context, "question": question})
        return response["output_text"]

def load_api_key_from_json(json_path="api_key.json"):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"API key file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)
        api_key = data.get("api_key")
        if not api_key:
            raise ValueError("File does not contain 'google_api_key'.")
        return api_key

# Global variable to hold the QA system instance
qa_system = None

# API endpoint to process the PDF and answer questions
@app.post("/process_pdf/")
async def process_pdf(file: UploadFile = File(...)):
    global qa_system
    try:
        # Load API key
        api_key = load_api_key_from_json("api_key.json")
        qa_system = PDFQuestionAnswering(api_key)

        # Read PDF file
        pdf_file = BytesIO(await file.read())
        text = qa_system.extract_pdf_text(pdf_file)
        chunks = qa_system.split_text(text)

        # Build vector store
        qa_system.vector_store = qa_system.build_vector_store(chunks)
        qa_system.chunks = chunks

        return {"message": "PDF processed successfully. You can now ask questions."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to check if the PDF is processed
@app.get("/pdf_status/")
async def pdf_status():
    # Check if the PDF is processed and ready for questions
    if qa_system and qa_system.vector_store:
        return {"status": "processed"}
    return {"status": "not_processed"}

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask_question/")
async def ask_question(request: QuestionRequest):
    try:
        if qa_system.vector_store is None:
            raise HTTPException(status_code=400, detail="PDF is not processed yet.")
        
        # Answer the question
        answer = qa_system.ask_question(request.question, qa_system.vector_store, qa_system.chunks)
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


