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