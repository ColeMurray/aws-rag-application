"""
FastAPI application for the RAG pipeline.

This module provides a REST API for querying documents using
Retrieval-Augmented Generation with AWS Bedrock and Pinecone.
"""
import json
import logging
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from pinecone import Pinecone
import structlog

from .config import get_settings, Settings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., min_length=1, max_length=1000, description="User question or query")
    max_chunks: Optional[int] = Field(default=None, ge=1, le=20, description="Maximum number of chunks to retrieve")
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum similarity score")
    include_sources: Optional[bool] = Field(default=True, description="Include source information in response")
    
    @field_validator('query')
    def validate_query(cls, v):
        """Validate and clean the query string."""
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class SourceInfo(BaseModel):
    """Source information for retrieved chunks."""
    source: str
    chunk_index: int
    similarity_score: float
    text_preview: str = Field(description="First 200 characters of the chunk")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    query: str
    sources: List[SourceInfo] = Field(default_factory=list)
    processing_time_ms: float
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]


class RAGService:
    """Service class for RAG operations."""
    
    def __init__(self, settings: Settings):
        """Initialize the RAG service with AWS and Pinecone clients."""
        self.settings = settings
        self._setup_clients()
        logger.info("RAG service initialized")
    
    def _setup_clients(self):
        """Initialize AWS Bedrock and Pinecone clients."""
        try:
            # Initialize Bedrock client
            self.bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=self.settings.aws_region
            )
            
            # Initialize Pinecone client
            self.pinecone_client = Pinecone(
                api_key=self.settings.pinecone_api_key,
                environment=self.settings.pinecone_environment
            )
            self.pinecone_index = self.pinecone_client.Index(self.settings.pinecone_index_name)
            
            logger.info(
                "Clients initialized successfully",
                region=self.settings.aws_region,
                index=self.settings.pinecone_index_name
            )
            
        except Exception as e:
            logger.error("Failed to initialize clients", error=str(e))
            raise
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Bedrock Titan.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.settings.embed_model_id,
                body=json.dumps({"inputText": text}),
                contentType="application/json",
                accept="application/json",
            )
            
            response_body = json.loads(response["body"].read())
            embedding = response_body["embedding"]
            
            logger.debug(
                "Generated embedding",
                text_length=len(text),
                embedding_dimension=len(embedding)
            )
            
            return embedding
            
        except ClientError as e:
            logger.error("AWS Bedrock error", error=str(e), model_id=self.settings.embed_model_id)
            raise HTTPException(
                status_code=503,
                detail=f"Embedding service unavailable: {e.response['Error']['Code']}"
            )
        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error during embedding generation")
    
    async def search_similar_chunks(self, query_embedding: List[float], top_k: int, threshold: float) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in Pinecone.
        
        Args:
            query_embedding: Query vector
            top_k: Number of chunks to retrieve
            threshold: Minimum similarity score
            
        Returns:
            List of matching chunks with metadata
        """
        try:
            # Query Pinecone index
            search_response = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Filter by similarity threshold
            filtered_matches = [
                match for match in search_response.matches
                if match.score >= threshold
            ]
            
            logger.info(
                "Similarity search completed",
                total_matches=len(search_response.matches),
                filtered_matches=len(filtered_matches),
                threshold=threshold
            )
            
            return filtered_matches
            
        except Exception as e:
            logger.error("Failed to search Pinecone", error=str(e))
            raise HTTPException(status_code=500, detail="Vector search service unavailable")
    
    async def generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer using Bedrock LLM.
        
        Args:
            query: User question
            context: Retrieved context from documents
            
        Returns:
            Generated answer
        """
        try:
            # Construct prompt
            prompt = (
                "You are a helpful assistant. Use the context delimited by <docs></docs> to answer the question. "
                "If the context doesn't contain enough information to answer the question, say so clearly. "
                "Provide specific details when available and cite relevant information from the context.\n\n"
                f"<docs>\n{context}\n</docs>\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )
            
            # Call Bedrock LLM with proper Anthropic format
            response = self.bedrock_client.invoke_model(
                modelId=self.settings.llm_model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.settings.max_tokens,
                    "temperature": self.settings.temperature,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ]
                }),
                contentType="application/json",
                accept="application/json",
            )
            
            response_body = json.loads(response["body"].read())
            
            # Handle Anthropic response format
            if "content" in response_body and response_body["content"]:
                answer = response_body["content"][0]["text"].strip()
            elif "outputs" in response_body:
                # Fallback for other formats
                answer = response_body["outputs"][0]["text"].strip()
            elif "completion" in response_body:
                # Alternative format
                answer = response_body["completion"].strip()
            else:
                # Fallback
                answer = str(response_body).strip()
            
            logger.debug(
                "Generated answer",
                query_length=len(query),
                context_length=len(context),
                answer_length=len(answer)
            )
            
            return answer
            
        except ClientError as e:
            logger.error("AWS Bedrock LLM error", error=str(e), model_id=self.settings.llm_model_id)
            raise HTTPException(
                status_code=503,
                detail=f"Language model service unavailable: {e.response['Error']['Code']}"
            )
        except Exception as e:
            logger.error("Failed to generate answer", error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error during answer generation")
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a RAG query end-to-end.
        
        Args:
            request: Query request with parameters
            
        Returns:
            Query response with answer and metadata
        """
        start_time = time.time()
        
        try:
            # Use request parameters or defaults from settings
            max_chunks = request.max_chunks or self.settings.top_k
            threshold = request.similarity_threshold or self.settings.similarity_threshold
            
            logger.info(
                "Processing query",
                query=request.query,
                max_chunks=max_chunks,
                threshold=threshold
            )
            
            # Step 1: Generate query embedding
            query_embedding = await self.generate_embedding(request.query)
            
            # Step 2: Search for similar chunks
            similar_chunks = await self.search_similar_chunks(query_embedding, max_chunks, threshold)
            
            if not similar_chunks:
                logger.warning("No similar chunks found", query=request.query, threshold=threshold)
                return QueryResponse(
                    answer="I couldn't find any relevant information to answer your question. Please try rephrasing your query or ask about a different topic.",
                    query=request.query,
                    sources=[],
                    processing_time_ms=round((time.time() - start_time) * 1000, 2),
                    timestamp=datetime.utcnow().isoformat(),
                    metadata={"chunks_found": 0, "threshold_used": threshold}
                )
            
            # Step 3: Prepare context from retrieved chunks
            context_parts = []
            sources = []
            
            for i, match in enumerate(similar_chunks):
                chunk_text = match.metadata.get("text", "")
                context_parts.append(f"[Source {i+1}]: {chunk_text}")
                
                if request.include_sources:
                    sources.append(SourceInfo(
                        source=match.metadata.get("source", "Unknown"),
                        chunk_index=match.metadata.get("chunk_index", 0),
                        similarity_score=round(match.score, 4),
                        text_preview=chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                    ))
            
            context = "\n\n".join(context_parts)
            
            # Step 4: Generate answer
            answer = await self.generate_answer(request.query, context)
            
            processing_time = round((time.time() - start_time) * 1000, 2)
            
            logger.info(
                "Query processed successfully",
                query=request.query,
                chunks_used=len(similar_chunks),
                processing_time_ms=processing_time
            )
            
            return QueryResponse(
                answer=answer,
                query=request.query,
                sources=sources,
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow().isoformat(),
                metadata={
                    "chunks_found": len(similar_chunks),
                    "threshold_used": threshold,
                    "model_used": self.settings.llm_model_id,
                    "embedding_model": self.settings.embed_model_id
                }
            )
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error("Unexpected error processing query", error=str(e), traceback=traceback.format_exc())
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def health_check(self) -> HealthResponse:
        """
        Perform health check on all services.
        
        Returns:
            Health status information
        """
        services = {}
        
        try:
            # Test Bedrock connection
            await self.generate_embedding("test")
            services["bedrock"] = "healthy"
        except Exception as e:
            logger.warning("Bedrock health check failed", error=str(e))
            services["bedrock"] = "unhealthy"
        
        try:
            # Test Pinecone connection
            self.pinecone_index.describe_index_stats()
            services["pinecone"] = "healthy"
        except Exception as e:
            logger.warning("Pinecone health check failed", error=str(e))
            services["pinecone"] = "unhealthy"
        
        overall_status = "healthy" if all(status == "healthy" for status in services.values()) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            version=self.settings.api_version,
            services=services
        )
    
    def _serialize_index_stats(self, index_stats) -> Dict[str, Any]:
        """
        Safely serialize Pinecone index stats to avoid JSON serialization issues.
        
        Args:
            index_stats: Pinecone index stats object
            
        Returns:
            Dictionary with serializable values
        """
        try:
            stats_dict = {
                "total_vector_count": getattr(index_stats, 'total_vector_count', 0),
                "dimension": getattr(index_stats, 'dimension', 0),
                "index_fullness": getattr(index_stats, 'index_fullness', 0.0),
            }
            
            # Handle namespaces safely
            namespaces = getattr(index_stats, 'namespaces', {})
            if namespaces:
                # Convert namespace objects to simple dictionaries
                namespace_dict = {}
                for ns_name, ns_obj in namespaces.items():
                    if hasattr(ns_obj, 'vector_count'):
                        namespace_dict[ns_name] = {
                            "vector_count": getattr(ns_obj, 'vector_count', 0)
                        }
                    else:
                        # Fallback for simple values
                        namespace_dict[ns_name] = str(ns_obj)
                stats_dict["namespaces"] = namespace_dict
            else:
                stats_dict["namespaces"] = {}
                
            return stats_dict
            
        except Exception as e:
            logger.warning("Failed to serialize index stats", error=str(e))
            return {"error": "Unable to serialize stats", "raw_type": str(type(index_stats))}


# Initialize FastAPI app
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


# Create app instance
app = create_app()

# Initialize RAG service
rag_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global rag_service
    try:
        settings = get_settings()
        rag_service = RAGService(settings)
        logger.info("Application startup completed")
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Application shutdown completed")


# Dependency to get RAG service
def get_rag_service() -> RAGService:
    """Dependency to get the RAG service instance."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return rag_service


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information."""
    settings = get_settings()
    return {
        "message": f"Welcome to {settings.api_title}",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(service: RAGService = Depends(get_rag_service)):
    """Health check endpoint."""
    return await service.health_check()


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    service: RAGService = Depends(get_rag_service)
):
    """
    Query documents using RAG.
    
    This endpoint processes natural language questions against the indexed documents
    and returns AI-generated answers with source citations.
    """
    response = await service.process_query(request)
    
    # Log query for analytics (in background)
    background_tasks.add_task(
        log_query_analytics,
        request.query,
        len(response.sources),
        response.processing_time_ms
    )
    
    return response


@app.get("/stats")
async def get_index_stats(service: RAGService = Depends(get_rag_service)):
    """Get Pinecone index statistics."""
    try:
        stats = service.pinecone_index.describe_index_stats()
        return {
            "index_stats": service._serialize_index_stats(stats),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Failed to get index stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve index statistics")


# Background task for analytics
def log_query_analytics(query: str, sources_count: int, processing_time: float):
    """Log query analytics for monitoring."""
    logger.info(
        "Query analytics",
        query_length=len(query),
        sources_returned=sources_count,
        processing_time_ms=processing_time,
        timestamp=datetime.utcnow().isoformat()
    )


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    logger.warning("Validation error", error=str(exc))
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "type": "validation_error"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors."""
    logger.error("Unhandled exception", error=str(exc), traceback=traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "server_error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    ) 