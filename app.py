import streamlit as st
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1) MUST be first
st.set_page_config(page_title="PDF‑RAG Chatbot", layout="centered")

st.title("📄 RAG Chatbot (PDF‑based)")
st.write("Streamlit + FAISS + OpenAI·GPT‑3.5")

# 2) Load & parse PDF once
@st.cache_resource
def load_vectorstore():
    # read PDF
    reader = PdfReader(os.path.join("data", "document.pdf"))
    full_text = "".join(p.extract_text() or "" for p in reader.pages)
    # chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_text(full_text)
    # embeddings
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    # build FAISS
    return FAISS.from_texts(docs, embedding)

vectordb = load_vectorstore()
retriever = vectordb.as_retriever()

# 3) LLM setup (requires streamlit secret OPENAI_API_KEY)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=st.secrets["OPENAI_API_KEY"]
)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 4) User query
query = st.chat_input("Ask a question about the PDF…")
if query:
    with st.spinner("Thinking…"):
        answer = qa.run(query)
    st.chat_message("user").write(query)
    st.chat_message("assistant").write(answer)
