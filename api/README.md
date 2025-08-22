# OpenAI Chat API Backend

This is a FastAPI-based backend with Retrieval-Augmented Generation (RAG), including classic vector search, RAG-Fusion, optional Tavily web search, and optional LangSmith tracing. It also provides a streaming chat interface using OpenAI's API.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- An OpenAI API key
- (Optional) A Tavily API key for web search
- (Optional) A LangSmith API key for tracing

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

1. Make sure you're in the `api` directory:
```bash
cd api
```

2. Start the server:
```bash
python app.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### Chat Endpoint
- **URL**: `/api/chat`
- **Method**: POST
- **Request Body**:
```json
{
    "developer_message": "string",
    "user_message": "string",
    "model": "gpt-4.1-mini",  // optional
    "api_key": "your-openai-api-key"
}
```
- **Response**: Streaming text response

### RAG Endpoints

- **Upload PDFs**
  - URL: `/api/rag/upload`
  - Method: POST (multipart/form-data)
  - Fields: `files` (one or more PDFs), `api_key` (OpenAI key)
  - Response: JSON with processing stats

- **RAG Chat (vector DB only)**
  - URL: `/api/rag/chat`
  - Method: POST (application/json)
  - Body:
  ```json
  {
    "user_message": "string",
    "model": "gpt-4.1-mini",
    "api_key": "your-openai-api-key",
    "k": 3
  }
  ```
  - Response: JSON with `response`, `sources`

- **RAG-Fusion Chat (with optional web search)**
  - URL: `/api/rag/fusion_chat`
  - Method: POST (application/json)
  - Body:
  ```json
  {
    "user_message": "string",
    "model": "gpt-4.1-mini",
    "api_key": "your-openai-api-key",
    "k": 5,
    "num_queries": 4,
    "include_web": true,
    "web_results": 3
  }
  ```
  - Behavior: Expands the user query into multiple reformulations, retrieves per-query results, fuses rankings via RRF, optionally appends Tavily web snippets, and generates the final answer.
  - Response: JSON with `response`, `sources`, and `fusion` metadata

### Health Check
- **URL**: `/api/health`
- **Method**: GET
- **Response**: `{"status": "ok"}`

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## CORS Configuration

The API is configured to accept requests from any origin (`*`). This can be modified in the `app.py` file if you need to restrict access to specific domains.

## Error Handling

The API includes basic error handling for:
- Invalid API keys
- OpenAI API errors
- General server errors

All errors will return a 500 status code with an error message. 

## Environment Variables

Set these to unlock optional goodies:

- `OPENAI_API_KEY`: Required for embeddings and chat
- `TAVILY_API_KEY`: Optional, enables web search snippets in fusion mode
- `LANGSMITH_API_KEY` (or `LANGCHAIN_API_KEY`): Optional, enables tracing for decorated pipeline steps
- `LANGCHAIN_TRACING_V2=true` (optional): Turn on LangSmith tracing

Pro tip: use a local `.env` file at the project root for dev.