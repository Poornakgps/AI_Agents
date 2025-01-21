import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
import google.generativeai as genai

from pathlib import Path
import tempfile
from dotenv import load_dotenv
import os

import pandas as pd

# Libraries for document reading
from PyPDF2 import PdfReader
from docx import Document

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Page configuration
st.set_page_config(
    page_title="Multimodal AI Agent - Document Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("Phidata Document AI Analyzer Agent 📄🖋️")
st.header("Powered by Gemini 2.0 Flash Exp")

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Document AI Analyzer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

# Initialize the agent
document_agent = initialize_agent()

# Persistent state for document and chat history
if "document_text" not in st.session_state:
    st.session_state.document_text = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "show_full_history" not in st.session_state:
    st.session_state.show_full_history = False

# Function to extract text from documents
def extract_text(file_path, file_type):
    try:
        if file_type == "pdf":
            reader = PdfReader(file_path)
            text = " ".join(page.extract_text() for page in reader.pages)
        elif file_type == "docx":
            doc = Document(file_path)
            text = " ".join(paragraph.text for paragraph in doc.paragraphs)
        elif file_type == "txt":
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
        else:
            text = None
        return text
    except Exception as e:
        st.error(f"Error reading the document: {e}")
        return None

# File uploader for documents
document_file = st.file_uploader(
    "Upload a document file", type=['pdf', 'docx', 'txt'], help="Upload a document for AI analysis"
)

# Detect file update and trigger a rerun
if document_file:
    file_type = document_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_doc:
        temp_doc.write(document_file.read())
        document_path = temp_doc.name

    st.info(f"Uploaded document: **{document_file.name}**")

    if st.session_state.document_text is None:
        if st.button("🔍 Analyze Document", key="analyze_document_button"):
            try:
                with st.spinner("Extracting text and analyzing document..."):
                    # Extract text from the document
                    document_text = extract_text(document_path, file_type)
                    if document_text:
                        st.session_state.document_text = document_text
                        st.success("Document analysis completed. You can now chat with the AI agent!")
                    else:
                        st.error("Failed to extract text from the document.")
            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
            finally:
                # Clean up temporary document file
                Path(document_path).unlink(missing_ok=True)
    else:
        st.info("Document has already been analyzed. Start chatting with the agent below.")

    # Chat interface
    user_query = st.text_area(
        "Chat with the AI Agent",
        placeholder="Ask follow-up questions or explore insights from the document.",
        help="Provide specific questions or insights you want from the document."
    )

    if st.button("💬 Send Query", key="send_query_button"):
        if not user_query:
            st.warning("Please enter a question to chat with the agent.")
        else:
            try:
                with st.spinner("AI Agent is responding..."):
                    # Append chat history
                    chat_prompt = (
                        f"""
                        The document content is as follows:
                        {st.session_state.document_text}

                        Respond to the following user query based on the document and context:
                        {user_query}
                        """
                    )
                    response = document_agent.run(chat_prompt)
                    st.session_state.chat_history.append(("User", user_query))
                    st.session_state.chat_history.append(("Agent", response.content))

                # Display only the most recent interaction
                st.subheader("Latest Interaction")
                st.markdown(f"**You:** {user_query}")
                st.markdown(f"**Agent:** {response.content}")

            except Exception as error:
                st.error(f"An error occurred during chat: {error}")

    # Display chat history
    st.subheader("Chat History")
    if st.session_state.chat_history:
        for role, message in st.session_state.chat_history:
            if role == "User":
                st.markdown(f"**You:** {message}")
            else:
                st.markdown(f"**Agent:** {message}")

        if st.button("View Full Chat History", key="view_chat"):
            st.session_state.show_full_history = not st.session_state.show_full_history

        if st.session_state.show_full_history:
            full_chat = "\n".join([f"{role}: {message}" for role, message in st.session_state.chat_history])
            st.text_area("Full Chat History", full_chat, height=300)

    # Download chat as Excel option
if st.session_state.chat_history:
    # Prepare data for Excel
    user_messages = []
    agent_responses = []

    for role, message in st.session_state.chat_history:
        if role == "User":
            user_messages.append(message)
            agent_responses.append("")  # Placeholder for alignment
        elif role == "Agent":
            if len(agent_responses) > len(user_messages):
                user_messages.append("")  # Align with the agent's response
            agent_responses[-1] = message  # Replace the placeholder with the actual response

    # Create DataFrame
    df = pd.DataFrame({"User": user_messages, "Agent": agent_responses})

    # Convert DataFrame to Excel
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as excel_file:
        df.to_excel(excel_file.name, index=False)
        excel_file_path = excel_file.name

    # Provide download button
    with open(excel_file_path, "rb") as f:
        st.download_button(
            label="📥 Download Chat History (Excel)",
            data=f.read(),
            file_name="chat_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
else:
    st.info("Upload a document file to begin analysis.")

# Customize text area height
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
