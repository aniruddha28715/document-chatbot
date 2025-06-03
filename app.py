import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Set page config
st.set_page_config(page_title="Document Chatbot", page_icon="📚")

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False

# Title and description
st.title("📚 Document Chatbot")
st.write("Upload a document and ask questions about its content!")

# File uploader
uploaded_file = st.file_uploader("Upload your document", type=['txt', 'pdf', 'docx'])

def process_document(file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    # Load document based on file type
    if file.name.endswith('.txt'):
        loader = TextLoader(tmp_file_path)
    elif file.name.endswith('.pdf'):
        loader = PyPDFLoader(tmp_file_path)
    elif file.name.endswith('.docx'):
        loader = Docx2txtLoader(tmp_file_path)
    else:
        st.error("Unsupported file type!")
        return None

    # Load and split the document
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    # Create vector store
    embeddings = OllamaEmbeddings(model="llama3.2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Create conversation chain
    llm = Ollama(model="llama3.2")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    # Clean up temporary file
    os.unlink(tmp_file_path)
    
    return conversation

# Process document when uploaded
if uploaded_file and not st.session_state.processed_docs:
    with st.spinner("Processing document..."):
        st.session_state.conversation = process_document(uploaded_file)
        st.session_state.processed_docs = True
        st.success("Document processed! You can now ask questions.")

# Chat interface
if st.session_state.conversation:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"question": prompt})
                st.write(response["answer"])
                st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
else:
    st.info("Please upload a document to start chatting!") 