from langchain_google_genai import ChatGoogleGenerativeAI
import os 
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import initialize_agent, Tool
from typing import List
import sqlite3
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import Optional
from pydantic import BaseModel, Field
from langchain.memory import ConversationEntityMemory , ConversationBufferWindowMemory
import asyncio
import threading
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import MessagesPlaceholder

load_dotenv()



if threading.current_thread() is not threading.main_thread():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


api_key=os.getenv("GOOGLE_API_KEY")


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash" , google_api_key=api_key)

#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history" , return_messages=True)
    

embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001" , google_api_key=api_key)

project_root = os.path.dirname(os.path.abspath(__file__)) 
faiss_path = os.path.abspath(os.path.join(project_root, "../faisa_api"))

faiss_index = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)




class Employee_data(BaseModel):
    employee_name: str = Field(description="name of the employee")
    request : str = Field(description="resone for requesting")
    description: Optional[str] = Field(
        default=None, description="description of the request"
    )
    start_date : str = Field(description="starting date of the leave")
    end_date : str = Field(description="requested leave last date")


structured_llm = llm.with_structured_output(Employee_data)

promtTemplate = ChatPromptTemplate([
    ("system", """you are HR chat Assistent , can u give the user leave request in the json format ,
                     if any required field is not given by the user , you don't hallucinate , please return
                     the user proper output meaasge"""),
    ("human", "{user_input}"),
])



system_prompt = (
    """You are an HR assistant. You can do two things:
    1. Answer employee questions about HR policies using the "decode_user_context" tool, 
     which retrieves relevant policy context from the knowledge base.
     Only answer questions using the provided context. If the context is insufficient, reply: 'I do not know.
     2. Generate reminder emails to employees who have taken more leaves than allowed. 
     Use the "sql_leave_fetch" tool, which fetches employees exceeding the leave limit. 
     For each employee, generate an email using this template:
     3.we provide a text data and to convert that into jaon and return it , for that use the "textToJson" tool
    
     To: [email address]
     Subject: Reminder: Leave Limit Exceeded
     Body:
     Dear [Employee Name],
    ... [email body] ...
   
    "Be friendly, professional, and informative in your emails."""
)

custom_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),           
    MessagesPlaceholder(variable_name="chat_history"),  
    ("human", "{input}"),    
])


@tool
def decode_user_context(query:str) -> str:
    """
        Use this tool to get relevant HR policy context based on query.
    """
    docs = faiss_index.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in docs])
    return context



@tool
def sql_leave_fetch( leave_days : int) -> List[str]:
    """
    Fetch employees from the SQLite database who have taken more than `leave_days` leaves.
    Returns a list of formatted strings like: "Alice Johnson (alice@company.com) - 8 leaves"
    """
    #db= "venv/employees.db"
    project_root = os.path.dirname(os.path.abspath(__file__))  # This will be "venv/"
    db_path = os.path.abspath(os.path.join(project_root, "../employees.db"))

    conn = sqlite3.connect(db_path)
    #conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name, email, total_leaves_taken 
        FROM employee_leaves
        WHERE total_leaves_taken > ?
    """, (leave_days,))
    
    rows = cursor.fetchall()
    conn.close()

    return [{"name": name, "email": email, "leave_days": leaves} for name, email, leaves in rows]


@tool
def textToJson(query : str) -> Employee_data:
    """
        covert the following text query to the json format and return that format to the llm
    """
    chain = promtTemplate | structured_llm
    response = chain.invoke({"user_input": query})
    return response
    

tools = [decode_user_context , sql_leave_fetch ,textToJson]


agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-conversational-react-description",
    handle_parsing_errors=True,
    verbose=True,
    agent_kwargs={"system_message": system_prompt},
    memory=memory
    )

def hr_assistant_bot(user_input: str):
    response = agent.invoke({"input": user_input})
    #print(memory)
    #print(memory.load_memory_variables({"input": "What are the data security methods this company uses?"}))
    return response['output']

#print(hr_assistant_bot("What are the data security methods this company uses?"))

#print(hr_assistant_bot("Generate mails for all employees who took more than 5 days leave this quarter."))
#print(hr_assistant_bot("dear ma'am , i am Kushal ,  i want leave from because of my sister marriage can u please aprove my leave request , convert these into json format"))


#print(hr_assistant_bot("generate mail for not approval of leave because of the late leave request"))


