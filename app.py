import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.title("🤖Hi I am ClimaMind a Context-Aware Chatbot (Groq + RAG)")
st.write("Ask questions regarding Global Warming. Powered by Groq and LangChain.")   
# st.write("API Key:", GROQ_API_KEY)


# 1️⃣ Load Document
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

# 2️⃣ Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 3️⃣ Create Embeddings (FREE - HuggingFace)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4️⃣ Create Vector Store
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# 5️⃣ Conversation Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 6️⃣ Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b",  # fast + free tier
    temperature=0.3
)

# 7️⃣ Create Conversational RAG Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Chat Input
user_question = st.text_input("Ask your question:")

if user_question:
    response = qa_chain.invoke({"question": user_question})
    st.write("### 🤖 Answer:")
    st.write(response["answer"])
