import os
import sys
import json
from argparse import ArgumentParser
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

class PDFQuestionAnswering:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API key is missing. Please provide a valid API key.")
        self.api_key = api_key
        genai.configure(api_key=self.api_key)

    def extract_pdf_text(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
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
        chain = load_qa_chain(model, chain_type="map_reduce")  # Use default prompt for map_reduce
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

def main():
    parser = ArgumentParser(description="PDF Question Answering System")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("api_key_file", nargs="?", default="api_key.json", help="API key JSON file path")
    args = parser.parse_args()

    try:
        api_key = load_api_key_from_json(args.api_key_file)
        qa_system = PDFQuestionAnswering(api_key)

        print("Processing the PDF...")
        text = qa_system.extract_pdf_text(args.pdf_path)
        chunks = qa_system.split_text(text)

        vector_store = qa_system.build_vector_store(chunks)

        print("PDF processed. You can now ask questions. Type 'stop' to end.")
        
        while True:
            question = input("Ask a question: ")
            if question.strip().lower() == "stop":
                print("Stopping the Q&A session...")
                break

            print("Answering the question...")
            answer = qa_system.ask_question(question, vector_store, chunks)
            print(f"Answer: {answer}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
