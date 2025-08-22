# Import required FastAPI components for building the API
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
# Import Pydantic for data validation and settings management
from pydantic import BaseModel
# Import OpenAI client for interacting with OpenAI's API
from openai import OpenAI
import os
import sys
from typing import Optional, List, Dict

# Add the parent directory to Python path to import aimakerspace
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aimakerspace import RAGPipeline

# Initialize FastAPI application with a title
app = FastAPI(title="OpenAI Chat API with RAG")

# Configure CORS (Cross-Origin Resource Sharing) middleware
# This allows the API to be accessed from different domains/origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin
    allow_credentials=True,  # Allows cookies to be included in requests
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers in requests
)

# In-memory storage for RAG pipelines (keyed by API key for simple isolation)
rag_pipelines: Dict[str, RAGPipeline] = {}

# Define the data model for chat requests using Pydantic
# This ensures incoming request data is properly validated
class ChatRequest(BaseModel):
    developer_message: Optional[str] = None  # Optional system/developer message
    user_message: Optional[str] = None       # Latest user message (for backward compat)
    messages: Optional[List[Dict[str, str]]] = None  # Full chat history if provided
    model: Optional[str] = "gpt-4.1-mini"   # Model name
    api_key: str                             # OpenAI API key

class RAGChatRequest(BaseModel):
    user_message: str                        # User question for RAG
    model: Optional[str] = "gpt-4.1-mini"   # Model name
    api_key: str                             # OpenAI API key
    k: Optional[int] = 3                     # Number of chunks to retrieve

class FusionChatRequest(BaseModel):
    user_message: str                        # User question
    model: Optional[str] = "gpt-4.1-mini"   # Model for final generation
    api_key: str                             # OpenAI API key
    k: Optional[int] = 5                     # Number of fused chunks to use
    num_queries: Optional[int] = 4           # Number of query reformulations
    include_web: Optional[bool] = False      # Include Tavily web snippets
    web_results: Optional[int] = 3           # How many web snippets to include

def get_or_create_rag_pipeline(api_key: str) -> RAGPipeline:
    """Get existing RAG pipeline or create new one for the API key"""
    if api_key not in rag_pipelines:
        rag_pipelines[api_key] = RAGPipeline(api_key)
    return rag_pipelines[api_key]

# Define the main chat endpoint that handles POST requests
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # Initialize OpenAI client with the provided API key
        client = OpenAI(api_key=request.api_key)
        
        # Build message list (backwards compatible)
        if request.messages:
            msg_payload = request.messages
        else:
            msg_payload = [
                {"role": "system", "content": request.developer_message or ""},
                {"role": "user", "content": request.user_message or ""},
            ]
        
        # Create an async generator function for streaming responses
        async def generate():
            try:
                stream = client.chat.completions.create(
                    model=request.model,
                    messages=msg_payload,
                    stream=True
                )
            except Exception as e:
                yield f"Error: {str(e)}"
                return
            
            # Yield each chunk of the response as it becomes available
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        # Return a streaming response to the client
        return StreamingResponse(generate(), media_type="text/plain")
    
    except Exception as e:
        # Handle any errors that occur during processing
        raise HTTPException(status_code=500, detail=str(e))

# RAG Endpoints

@app.post("/api/rag/upload")
async def upload_pdf(files: List[UploadFile] = File(...), api_key: str = Form(...)):
    """Upload and process multiple PDFs for RAG"""
    try:
        print(f"Upload request received - Files: {len(files) if files else 0}, API key provided: {bool(api_key)}")
        
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        
        if not files:
            raise HTTPException(status_code=400, detail="At least one PDF file is required")
        
        # Validate all files are PDFs
        for file in files:
            if not file.filename or not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename or 'unknown'} is not a PDF. Only PDF files are supported")
        
        print(f"Processing {len(files)} PDF files...")
        
        # Get or create RAG pipeline for this API key
        rag_pipeline = get_or_create_rag_pipeline(api_key)
        
        total_chunks_created = 0
        total_characters = 0
        successful_files = []
        failed_files = []
        
        # Process each PDF file
        for i, file in enumerate(files):
            try:
                print(f"Processing file {i+1}/{len(files)}: {file.filename}")
                
                # Read PDF content
                pdf_content = await file.read()
                print(f"Read {len(pdf_content)} bytes from {file.filename}")
                
                # Process PDF
                result = await rag_pipeline.add_pdf(file.filename, pdf_content)
                print(f"Processing result for {file.filename}: {result}")
                
                if result["status"] == "success":
                    successful_files.append(file.filename)
                    total_chunks_created += result["chunks_created"]
                    total_characters += result["total_characters"]
                else:
                    failed_files.append(f"{file.filename}: {result['message']}")
                    
            except Exception as e:
                error_msg = f"Error processing file {file.filename}: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                failed_files.append(error_msg)
        
        print(f"Upload processing complete - Success: {len(successful_files)}, Failed: {len(failed_files)}")
        
        # Prepare response
        if successful_files and not failed_files:
            response = {
                "status": "success",
                "message": f"Successfully processed {len(successful_files)} PDF(s)",
                "successful_files": successful_files,
                "total_chunks_created": total_chunks_created,
                "total_characters": total_characters
            }
            print(f"Returning success response: {response}")
            return response
        elif successful_files and failed_files:
            response = {
                "status": "partial_success",
                "message": f"Processed {len(successful_files)} PDF(s) successfully, {len(failed_files)} failed",
                "successful_files": successful_files,
                "failed_files": failed_files,
                "total_chunks_created": total_chunks_created,
                "total_characters": total_characters
            }
            print(f"Returning partial success response: {response}")
            return response
        else:
            error_detail = f"All files failed to process: {'; '.join(failed_files)}"
            print(f"All files failed: {error_detail}")
            raise HTTPException(status_code=400, detail=error_detail)
        
    except HTTPException as he:
        print(f"HTTP Exception in upload: {he.status_code} - {he.detail}")
        raise
    except Exception as e:
        error_msg = f"Unexpected error processing PDFs: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/rag/chat")
async def rag_chat(request: RAGChatRequest):
    """Chat with documents using RAG"""
    try:
        # Get RAG pipeline for this API key
        if request.api_key not in rag_pipelines:
            raise HTTPException(status_code=400, detail="No documents uploaded for this API key")
        
        rag_pipeline = rag_pipelines[request.api_key]
        
        # Debug: Log the query
        print(f"RAG Query: {request.user_message}")
        
        # Search for relevant chunks
        context_chunks = rag_pipeline.search_documents(request.user_message, k=request.k)
        
        # Debug: Log the retrieved chunks
        print(f"Retrieved {len(context_chunks)} chunks:")
        for i, chunk in enumerate(context_chunks):
            print(f"  Chunk {i+1}: {chunk[:100]}...")
        
        if not context_chunks:
            raise HTTPException(status_code=400, detail="No relevant context found in uploaded documents")
        
        # Generate response
        response = rag_pipeline.generate_rag_response(
            request.user_message, 
            context_chunks, 
            model=request.model
        )
        
        return {
            "response": response,
            "context_chunks_used": len(context_chunks),
            "sources": [chunk[:100] + "..." for chunk in context_chunks]  # Preview of sources
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/fusion_chat")
async def rag_fusion_chat(request: FusionChatRequest):
    """Chat using RAG-Fusion with optional web augmentation."""
    try:
        # Get or create pipeline for this API key
        rag_pipeline = get_or_create_rag_pipeline(request.api_key)

        print(f"RAG-Fusion Query: {request.user_message}")

        # Retrieve fused chunks (optionally with web)
        context_chunks = rag_pipeline.rag_fusion(
            query=request.user_message,
            k=request.k or 5,
            num_queries=request.num_queries or 4,
            include_web=bool(request.include_web),
            web_results=request.web_results or 3,
        )

        if not context_chunks:
            raise HTTPException(status_code=400, detail="No relevant context found across fusion sources")

        # Generate final answer
        response = rag_pipeline.generate_rag_response(
            request.user_message,
            context_chunks,
            model=request.model or "gpt-4.1-mini",
        )

        return {
            "response": response,
            "context_chunks_used": len(context_chunks),
            "sources": [chunk[:160] + "..." for chunk in context_chunks],  # Slightly longer preview
            "fusion": {
                "num_queries": request.num_queries or 4,
                "include_web": bool(request.include_web),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rag/documents")
async def get_documents(api_key: str):
    """Get information about uploaded documents"""
    if api_key not in rag_pipelines:
        return {"loaded_documents": [], "total_chunks": 0, "vector_count": 0}
    
    rag_pipeline = rag_pipelines[api_key]
    return rag_pipeline.get_document_info()

@app.delete("/api/rag/documents")
async def clear_documents(api_key: str):
    """Clear all uploaded documents for an API key"""
    if api_key in rag_pipelines:
        rag_pipelines[api_key].clear_documents()
        return {"message": "All documents cleared successfully"}
    return {"message": "No documents found for this API key"}

# Define a health check endpoint to verify API status
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

# Test endpoint for debugging
@app.get("/api/test")
async def test_endpoint():
    return {"message": "API is working", "timestamp": "2025-01-03"}

@app.post("/api/test-upload")
async def test_upload(files: List[UploadFile] = File(...), api_key: str = Form(...)):
    """Test upload endpoint for debugging"""
    return {
        "files_received": len(files),
        "file_names": [f.filename for f in files],
        "api_key_provided": bool(api_key),
        "status": "test_success"
    }

# Entry point for running the application directly
if __name__ == "__main__":
    import uvicorn
    # Start the server on all network interfaces (0.0.0.0) on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
