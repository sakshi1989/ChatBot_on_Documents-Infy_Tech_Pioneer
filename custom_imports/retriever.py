import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever

# Load the .env file
load_dotenv()

# Read the stored embeddings from the vectorstore
embeddings = OpenAIEmbeddings()

vectorstore_db = FAISS.load_local("./vectorstore", embeddings)

retriever = vectorstore_db.as_retriever(search_type="similarity_score_threshold",
                                         search_kwargs={"score_threshold": 0.65})