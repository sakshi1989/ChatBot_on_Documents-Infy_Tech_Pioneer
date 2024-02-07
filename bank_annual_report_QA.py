import streamlit as st
import openai
from datetime import datetime
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Configures the default settings of the page.
st.set_page_config(page_title="2022BankAnnualReportChatbot",
                   page_icon=":robot_face:", layout="centered", initial_sidebar_state="auto")

# Display text in title formatting.
st.title(":blue[Welcome to the 2022 Banks Annual Report ChatBot] :robot_face:")

# Path of the bank logos
# To support the static media embedding that do not not work with normal media elements
# enable the feature "STREAMLIT_SERVER_ENABLE_STATIC_SERVING" in the .streamlit/config.toml file.
# Also store the images in the static folder and access it using app/static/<image_name>
# Ref:- https://docs.streamlit.io/library/advanced-features/static-file-serving
bofa_logo_path = 'app/static/bofa_logo.png'
wf_logo_path = 'app/static/wf_logo.png'

# Display the image inline with text using HTML and CSS
st.markdown(f"""
<div>    
    <p>Ask me anything about the Wells Fargo <img src="{wf_logo_path}" width="30" height="30"> & 
    Bank of America <img src="{bofa_logo_path}" width="40" height="40">  2022 Annual Report</p>
</div>
""", unsafe_allow_html=True)

# Get an OpenAI API Key before continuing
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Enter an OpenAI API Key to continue")
    st.stop()
else:
    # Check the validity of the API Key
    client = openai.OpenAI(api_key=openai_api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        st.info("Invalid API Key")
        st.stop()     


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(
            content="How can I help you today?",
            time=datetime.now(),
        ),
    ]

# Initialize the vectorstore and retriever
# Read the stored embeddings from the vectorstore
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

vectorstore_db = FAISS.load_local("./vectorstore", embeddings)

retriever = vectorstore_db.as_retriever(search_type="similarity_score_threshold",
                                        search_kwargs={"score_threshold": 0.65})

# Define the llm model
llm = ChatOpenAI(temperature=0.3, model='gpt-3.5-turbo-16k',
                openai_api_key=openai_api_key)

# Set up the context based LLMChain, passing in memory
contextualize_ques_system_prompt = """Given a chat history and the latest user question \
which might refer to a context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is. \
"""

contextualize_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_ques_system_prompt),
        # Prompt template that assumes variable is already list of messages.
        # We provide the variable name to be used as messages
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_ques_chain = contextualize_question_prompt | llm | StrOutputParser()

# The context parameter is the placeholder for the documents retrieved by the retriever & the
# contextualized question created
qa_system_prompt = """You are an assistant for question-answering tasks.
Answer the user's question based on the information in the documents and the provided context.

You always follow these guidelines:
    -If the answer isn't available within the documents, say "I don't know" 
    -Do not give answers from your knowledge

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_ques_chain
    else:
        return input["question"]


rag_chain = (
    RunnablePassthrough.assign(
        context=contextualized_question | retriever
    )
    | qa_prompt
    | llm
)

# Render current messages from StreamlitChatMessageHistory
for msg in st.session_state.messages:
    st.chat_message(msg.type, avatar="🧑" if msg.type == 'human' else "🤖").write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message(name="human", avatar="🧑").write(prompt)

    # Generate a response from AI
    response = None
    with st.spinner("Thinking..."):
        response = rag_chain.invoke(
            {"question": prompt, "chat_history": st.session_state.messages})
        print(response)
        st.chat_message(name="ai", avatar="🤖").write(response.content)

    human_message = HumanMessage(content=prompt, time=datetime.now())
    ai_message = AIMessage(content=response.content, time=datetime.now())

    # Store the messages in the session state (maintain history)
    st.session_state.messages.extend([human_message, ai_message])
