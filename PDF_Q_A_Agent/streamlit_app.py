import streamlit as st
import requests

# FastAPI backend URL
API_URL_PROCESS = "http://127.0.0.1:8000/process_pdf/"
API_URL_STATUS = "http://127.0.0.1:8000/pdf_status/"
API_URL_ASK = "http://127.0.0.1:8000/ask_question/"

def main():
    st.title("PDF Question Answering System")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.write("File uploaded successfully!")

        # Call backend to process the PDF
        files = {'file': uploaded_file.getvalue()}
        response = requests.post(API_URL_PROCESS, files=files)

        if response.status_code == 200:
            st.write("PDF processing completed. You can now ask questions.")
        else:
            st.write("Error processing the PDF.")

        # Check PDF processing status
        status_response = requests.get(API_URL_STATUS)
        if status_response.status_code == 200:
            status = status_response.json().get("status")
            if status == "processed":
                # Ask a question
                question = st.text_input("Ask a question about the PDF")

                if question:
                    st.write("Processing your question...")
                    question_response = requests.post(API_URL_ASK, json={"question": question})

                    if question_response.status_code == 200:
                        answer = question_response.json().get("answer")
                        st.write(f"Answer: {answer}")
                    else:
                        st.write("Error: Unable to process the question.")
            else:
                st.write("PDF is still being processed. Please wait.")
        else:
            st.write("Error checking PDF status.")

if __name__ == "__main__":
    main()
