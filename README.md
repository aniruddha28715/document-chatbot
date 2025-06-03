# Document Chatbot with Streamlit and Ollama

A powerful document-based chatbot that allows you to upload documents (PDF, TXT, DOCX) and ask questions about their content. The application uses Streamlit for the user interface and Ollama for local LLM processing.

## 🌟 Features

- **Multiple Document Formats**: Support for PDF, TXT, and DOCX files
- **Local LLM Processing**: Uses Ollama's llama3.2 model for processing
- **Real-time Processing**: Instant document analysis and question answering
- **Chat History**: Maintains conversation context during the session
- **User-friendly Interface**: Clean and intuitive Streamlit interface
- **Vector-based Search**: Efficient document retrieval using Chroma vector store
- **Memory Management**: Maintains conversation context using ConversationBufferMemory

## 📚 Code Structure and Explanation

### 1. Main Application (`app.py`)

#### Imports and Setup
```python
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
```
- **Streamlit**: Web framework for the user interface
- **LangChain**: Framework for document processing and LLM interactions
- **Chroma**: Vector store for document embeddings
- **Ollama**: Local LLM integration

#### Session State Management
```python
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False
```
- Maintains conversation state across page refreshes
- Tracks chat history and document processing status

#### Document Processing Function
```python
def process_document(file):
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    # Load document based on type
    if file.name.endswith('.txt'):
        loader = TextLoader(tmp_file_path)
    elif file.name.endswith('.pdf'):
        loader = PyPDFLoader(tmp_file_path)
    elif file.name.endswith('.docx'):
        loader = Docx2txtLoader(tmp_file_path)

    # Split document into chunks
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    # Create vector store and conversation chain
    embeddings = OllamaEmbeddings(model="llama3.2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
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

    # Cleanup
    os.unlink(tmp_file_path)
    return conversation
```
This function:
1. Creates a temporary file for the uploaded document
2. Loads the document based on its type
3. Splits the document into manageable chunks
4. Creates embeddings and vector store
5. Sets up the conversation chain with memory
6. Cleans up temporary files

#### User Interface Components
```python
# File uploader
uploaded_file = st.file_uploader("Upload your document", type=['txt', 'pdf', 'docx'])

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
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response["answer"]}
                )
```
The UI includes:
1. Document upload interface
2. Processing status indicators
3. Chat history display
4. Interactive chat input
5. Response generation with loading indicators

### 2. Dependencies (`requirements.txt`)
```txt
streamlit==1.32.0
python-docx==1.1.0
PyPDF2==3.0.1
pypdf==3.17.4
langchain==0.1.12
langchain-community>=0.0.28
chromadb==0.4.24
sentence-transformers==2.5.1
docx2txt==0.8
```
Each dependency serves a specific purpose:
- `streamlit`: Web interface
- `python-docx` & `docx2txt`: DOCX file processing
- `PyPDF2` & `pypdf`: PDF file processing
- `langchain`: Document processing framework
- `chromadb`: Vector storage
- `sentence-transformers`: Text embeddings

## 🔄 Application Flow

1. **Document Upload**
   - User uploads a document
   - System validates file type
   - Document is processed and chunked

2. **Vector Store Creation**
   - Document chunks are converted to embeddings
   - Embeddings are stored in Chroma
   - Vector store is created for retrieval

3. **Question Answering**
   - User asks a question
   - System retrieves relevant document chunks
   - LLM generates response based on context
   - Response is displayed to user

4. **Memory Management**
   - Conversation history is maintained
   - Context is preserved for follow-up questions
   - Session state is managed by Streamlit

## 🚀 Prerequisites

Before you begin, ensure you have the following installed:

1. **Python 3.8 or higher**
   ```bash
   python --version
   ```

2. **Ollama**
   - Install Ollama from [ollama.ai](https://ollama.ai)
   - Pull the llama3.2 model:
     ```bash
     ollama pull llama3.2
     ```

3. **Git** (for cloning the repository)
   ```bash
   git --version
   ```

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/aniruddha28715/document-chatbot.git
   cd document-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Ollama server**
   ```bash
   ollama serve
   ```
   Note: Keep this running in a separate terminal window.

## 🏃‍♂️ Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - Open your web browser
   - Navigate to `http://localhost:8502`

## 💻 Usage

1. **Upload a Document**
   - Click the "Upload your document" button
   - Select a PDF, TXT, or DOCX file
   - Wait for the document to be processed

2. **Ask Questions**
   - Type your question in the chat input
   - Press Enter or click the send button
   - View the AI's response based on the document content

3. **Chat History**
   - Previous questions and answers are displayed in the chat interface
   - The conversation context is maintained throughout the session

## 🔧 Technical Details

### Architecture
- **Frontend**: Streamlit
- **LLM**: Ollama (llama3.2)
- **Vector Store**: Chroma
- **Document Processing**: LangChain
- **Embeddings**: OllamaEmbeddings

### Key Components
- `app.py`: Main application file
- `requirements.txt`: Python dependencies
- `README.md`: Documentation

## ⚠️ Important Notes

1. **Ollama Server**
   - Must be running for the application to work
   - Default port: 11434
   - If you see "address already in use" error, Ollama is already running

2. **System Requirements**
   - Minimum 8GB RAM recommended
   - Sufficient disk space for document processing
   - Stable internet connection for initial setup

3. **Document Limitations**
   - Maximum file size: 100MB
   - Supported formats: PDF, TXT, DOCX
   - Text-based documents work best

## 🔍 Troubleshooting

1. **Ollama Connection Issues**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   ```

2. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --upgrade
   ```

3. **Port Conflicts**
   - If port 8502 is in use, Streamlit will automatically use the next available port
   - Check the terminal output for the correct URL

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [Ollama](https://ollama.ai/) for the LLM capabilities
- [LangChain](https://python.langchain.com/) for the document processing framework 