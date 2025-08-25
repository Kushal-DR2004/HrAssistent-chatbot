import logging
import streamlit as st
from dotenv import load_dotenv
#from policy_QA import policy_question
#from app import emailGeneratore
#from emailGenerationagent import leave_balance
#from jsonoutputformat import textTojson

from agentType import hr_assistant_bot


from langchain_google_genai import ChatGoogleGenerativeAI
import os 
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio
import threading

load_dotenv()

if threading.current_thread() is not threading.main_thread():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

api_key=os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash"  , google_api_key=api_key )


embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001" , google_api_key=api_key)

project_root = os.path.dirname(os.path.abspath(__file__))  # This will be "venv/"
faiss_path = os.path.abspath(os.path.join(project_root, "../faisa_api"))

faiss_index = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

#faiss_index = FAISS.load_local("../faisa_api", embeddings , allow_dangerous_deserialization=True)

DISALLOWED_KEYWORDS = ["salary", "home address", "ssn", "bank account"]

def is_disallowed_query(text: str) -> bool:
    return any(word in text.lower() for word in DISALLOWED_KEYWORDS)



def log_interaction(user_input: str, response: str):
    logging.info(f"User: {user_input}")
    logging.info(f"Assistant: {response}")




if "history" not in st.session_state:
    st.session_state.history = []

def process_user_query(user_query):
    if is_disallowed_query(user_query):
        response = "Query Blocked"
    else:
        response = hr_assistant_bot(user_query)
    return response

st.title("HR Assistant Interface")

user_input = st.text_area("Enter your query:", height=100)

submit = st.button("Submit")

if submit and user_input:
    with st.spinner("Waiting for model response..."):
        response = process_user_query(user_input)
        st.session_state.history.append((user_input, response))


for user_msg, assistant_msg in st.session_state.history:
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**Assistant:** {assistant_msg}")
    st.markdown("---")


