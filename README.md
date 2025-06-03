# Document Chatbot

A Streamlit-based chatbot that allows you to upload documents (PDF, TXT, or DOCX) and ask questions about their content. The chatbot uses Ollama's LLM models to provide responses based solely on the uploaded document's content.

## Prerequisites

1. Python 3.8 or higher
2. Ollama installed on your system
3. The llama2 model pulled in Ollama

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running on your system and you have the llama2 model:
```bash
ollama pull llama2
```

## Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Usage

1. Upload a document (PDF, TXT, or DOCX) using the file uploader
2. Wait for the document to be processed
3. Start asking questions about the document's content
4. The chatbot will respond based only on the information present in the uploaded document

## Features

- Support for multiple document formats (PDF, TXT, DOCX)
- Real-time document processing
- Chat history preservation during the session
- Responses based solely on document content
- User-friendly interface

## Note

Make sure you have enough system resources as the document processing and LLM operations can be memory-intensive. 