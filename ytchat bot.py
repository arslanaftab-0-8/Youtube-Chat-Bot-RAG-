import os
import streamlit as st
import yt_dlp
import webvtt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import LangChainException

# --- App Configuration ---
st.set_page_config(
    page_title="YouTube Video Q&A with Gemini",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions (with Caching) ---

@st.cache_data(show_spinner="Downloading and processing video captions...")
def download_and_parse_transcript(video_url):
    """
    Downloads the captions for a YouTube video and returns the plain text.
    Caches the result to avoid re-downloading for the same URL.
    """
    try:
        # 1. Download VTT captions using yt-dlp
        video_id = video_url.split("v=")[-1].split("&")[0]
        output_vtt_file = f"{video_id}.en.vtt"
        
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'subtitlesformat': 'vtt',
            'outtmpl': f'{video_id}.%(ext)s',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        if not os.path.exists(output_vtt_file):
            st.error("Failed to download captions. The video might not have English captions available.")
            return None

        # 2. Convert VTT to plain text
        transcript = " ".join(caption.text for caption in webvtt.read(output_vtt_file))
        
        # Clean up the downloaded file
        os.remove(output_vtt_file)
        
        return transcript

    except Exception as e:
        st.error(f"An error occurred during caption download: {e}")
        return None

@st.cache_data(show_spinner="Creating vector store...")
def create_vector_store(transcript, api_key):
    """
    Splits transcript, creates embeddings, and builds a FAISS vector store.
    Caches the result to avoid re-computation for the same transcript.
    """
    if not transcript:
        return None
    try:
        # 1. Split transcript into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
        chunks = splitter.create_documents([transcript])

        # 2. Embed with Gemini
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        
        # 3. Create FAISS vector store
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    except Exception as e:
        st.error(f"Failed to create vector store. Check your API key. Error: {e}")
        return None

def get_rag_chain(retriever, llm):
    """
    Creates and returns the full RAG chain.
    """
    prompt_template = """
    You are a helpful assistant who answers questions based ONLY on the provided video transcript context.
    Your answers should be detailed and directly reference the information in the text.
    Do not use any outside knowledge.
    If the context is insufficient to answer the question, simply state: "The video transcript does not contain information on this topic."

    CONTEXT:
    {context}

    QUESTION: {question}

    ANSWER:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()
    return rag_chain_from_docs | prompt | llm | parser


# --- Streamlit UI ---

st.title("📺 YouTube Video Q&A with Gemini")
st.markdown("""
    Ask questions about a YouTube video! This app uses Google's Gemini models to analyze the video's transcript and answer your questions.
    
    **How to use:**
    1.  Enter your Google Gemini API key.
    2.  Paste the URL of the YouTube video.
    3.  Once the video is processed, ask your question!
""")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")
    gemini_api_key = st.text_input("Gemini API Key", type="password", help="Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey)")
    youtube_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

    process_button = st.button("Process Video")

# --- Main Content Area ---
if process_button:
    if not gemini_api_key:
        st.warning("Please enter your Gemini API key.")
    elif not youtube_url:
        st.warning("Please enter a YouTube URL.")
    else:
        # Set the API key for the session
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        # Store retriever in session state to persist it
        transcript = download_and_parse_transcript(youtube_url)
        if transcript:
            st.session_state.retriever = create_vector_store(transcript, gemini_api_key)
            st.success("Video processed successfully! You can now ask questions.")

if "retriever" in st.session_state and st.session_state.retriever:
    st.header("Ask a Question")
    
    question = st.text_input("e.g., 'What is the main topic of the video?'")

    if question:
        try:
            # Initialize the LLM
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=gemini_api_key)
            
            # Get and run the RAG chain
            rag_chain = get_rag_chain(st.session_state.retriever, llm)
            
            with st.spinner("Finding the answer..."):
                answer = rag_chain.invoke(question)
                st.markdown("### Answer")
                st.write(answer)

        except LangChainException as e:
            st.error(f"An error occurred while communicating with the Gemini API: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

