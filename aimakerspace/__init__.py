from .text_utils import TextFileLoader, CharacterTextSplitter, PDFLoader
from .vectordatabase import VectorDatabase
from .openai_utils.embedding import EmbeddingModel
from .openai_utils.chatmodel import ChatOpenAI

# RAG Pipeline
import numpy as np
from typing import List, Dict, Any, Optional
import asyncio
from openai import OpenAI, AsyncOpenAI


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for PDF documents
    """
    
    def __init__(self, api_key: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize RAG pipeline
        
        :param api_key: OpenAI API key
        :param chunk_size: Size of text chunks for processing
        :param chunk_overlap: Overlap between chunks
        """
        self.api_key = api_key
        
        # Initialize embedding model with API key
        self.embedding_model = EmbeddingModel(api_key=api_key)
        
        self.vector_db = VectorDatabase(embedding_model=self.embedding_model)
        self.text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.pdf_loader = PDFLoader()
        
        # Initialize chat model with API key
        self.chat_model = ChatOpenAI(api_key=api_key)
        
        # In-memory storage for documents
        self.documents: Dict[str, str] = {}  # filename -> full text
        self.chunks: Dict[str, List[str]] = {}  # filename -> list of chunks
        
    async def add_pdf(self, filename: str, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Add a PDF document to the RAG pipeline
        
        :param filename: Name of the PDF file
        :param pdf_bytes: PDF file content as bytes
        :return: Status information
        """
        try:
            # Extract text from PDF
            text = self.pdf_loader.load_from_bytes(pdf_bytes)
            
            if not text.strip():
                return {"status": "error", "message": "No text could be extracted from the PDF"}
            
            # Store full document
            self.documents[filename] = text
            
            # Split text into chunks
            chunks = self.text_splitter.split(text)
            self.chunks[filename] = chunks
            
            # Create embeddings and store in vector database
            chunk_keys = [f"{filename}__chunk_{i}" for i in range(len(chunks))]
            embeddings = await self.embedding_model.async_get_embeddings(chunks)
            
            for key, chunk, embedding in zip(chunk_keys, chunks, embeddings):
                self.vector_db.insert(key, np.array(embedding))
            
            return {
                "status": "success", 
                "message": f"Successfully processed {filename}",
                "chunks_created": len(chunks),
                "total_characters": len(text)
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Error processing PDF: {str(e)}"}
    
    def search_documents(self, query: str, k: int = 3) -> List[str]:
        """
        Search for relevant document chunks
        
        :param query: Search query
        :param k: Number of chunks to retrieve
        :return: List of relevant text chunks
        """
        # Increase k to get more potential matches, then we'll return the top k
        search_k = min(k * 2, len(self.vector_db.vectors))  # Get more candidates
        
        # Get search results with similarity scores
        results_with_scores = self.vector_db.search_by_text(query, k=search_k, return_as_text=False)
        
        # Debug: Print similarity scores
        print(f"Search results for '{query}':")
        for i, (chunk_key, score) in enumerate(results_with_scores[:k]):
            print(f"  Result {i+1}: Score={score:.4f}, Key={chunk_key}")
        
        # Extract just the text chunks (top k results)
        top_chunks = []
        for chunk_key, score in results_with_scores[:k]:
            # Get the actual chunk text from the key
            filename, chunk_info = chunk_key.split('__chunk_')
            chunk_idx = int(chunk_info)
            if filename in self.chunks and chunk_idx < len(self.chunks[filename]):
                top_chunks.append(self.chunks[filename][chunk_idx])
        
        return top_chunks
    
    def generate_rag_response(self, query: str, context_chunks: List[str], model: str = "gpt-4.1-mini") -> str:
        """
        Generate a response using retrieved context
        
        :param query: User query
        :param context_chunks: Retrieved context chunks
        :param model: Model to use for generation
        :return: Generated response
        """
        context = "\n\n".join(context_chunks)
        
        # Debug: Print the context being used
        print(f"Context being used for query '{query}':")
        print(f"Context length: {len(context)} characters")
        print(f"Context preview: {context[:200]}...")
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context. Use the information from the context to answer questions as best as you can. If the context contains relevant information, use it to provide a helpful answer. If the context doesn't contain enough information to fully answer the question, provide what information you can from the context and mention what additional information might be needed."
            },
            {
                "role": "user",
                "content": f"Based on the following context, please answer this question:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            }
        ]
        
        # Create a new OpenAI client with the API key for this request
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7  # Add some creativity while staying factual
        )
        
        return response.choices[0].message.content
    
    def get_document_info(self) -> Dict[str, Any]:
        """Get information about loaded documents"""
        return {
            "loaded_documents": list(self.documents.keys()),
            "total_chunks": sum(len(chunks) for chunks in self.chunks.values()),
            "vector_count": len(self.vector_db.vectors)
        }
    
    def clear_documents(self):
        """Clear all loaded documents and vectors"""
        self.documents.clear()
        self.chunks.clear()
        self.vector_db.vectors.clear()
