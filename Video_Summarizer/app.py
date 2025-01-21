import streamlit as st 
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai

import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Page configuration
st.set_page_config(
    page_title="Multimodal AI Agent - Video Summarizer",
    page_icon="🎥",
    layout="wide"
)

st.title("Phidata Video AI Summarizer Agent 🎥🎤🖬")
st.header("Powered by Gemini 2.0 Flash Exp")

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

# Initialize the agent
multimodal_Agent = initialize_agent()

# File uploader
video_file = st.file_uploader(
    "Upload a video file", type=['mp4', 'mov', 'avi'], help="Upload a video for AI analysis"
)

# Persistent state for video and chat history
if "processed_video" not in st.session_state:
    st.session_state.processed_video = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path, format="video/mp4", start_time=0)

    if st.session_state.processed_video is None:
        if st.button("🔍 Analyze Video", key="analyze_video_button"):
            try:
                with st.spinner("Processing video and gathering insights..."):
                    # Upload and process video file
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    st.session_state.processed_video = processed_video

                st.success("Video analysis completed. You can now chat with the AI agent!")
            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
            finally:
                # Clean up temporary video file
                Path(video_path).unlink(missing_ok=True)
    else:
        st.info("Video has already been analyzed. Start chatting with the agent below.")

    # Chat interface
    user_query = st.text_area(
        "Chat with the AI Agent",
        placeholder="Ask follow-up questions or explore insights from the video.",
        help="Provide specific questions or insights you want from the video."
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
                        Respond to the following user query based on the analyzed video and context:
                        {user_query}
                        """
                    )
                    response = multimodal_Agent.run(chat_prompt, videos=[st.session_state.processed_video])
                    st.session_state.chat_history.append({"user": user_query, "agent": response.content})

                # Display chat history dynamically
                st.subheader("Chat History")
                for chat in st.session_state.chat_history:
                    st.markdown(f"**You:** {chat['user']}")
                    st.markdown(f"**Agent:** {chat['agent']}")

            except Exception as error:
                st.error(f"An error occurred during chat: {error}")
else:
    st.info("Upload a video file to begin analysis.")

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
