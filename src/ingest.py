"""
Data ingestion script for the RAG pipeline.

This script processes documents from S3 or local storage, chunks them,
generates embeddings using AWS Bedrock, and stores them in Pinecone.
"""
import json
import logging
import pathlib
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime, UTC
import hashlib

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
import structlog

from .config import get_settings

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


class DocumentIngester:
    """Handles document ingestion into the RAG pipeline."""
    
    def __init__(self):
        """Initialize the ingester with AWS and Pinecone clients."""
        self.settings = get_settings()
        self._setup_clients()
        self._setup_text_splitter()
    
    def _setup_clients(self):
        """Initialize AWS Bedrock and Pinecone clients."""
        try:
            # Initialize Bedrock client
            self.bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=self.settings.aws_region
            )
            logger.info("Bedrock client initialized", region=self.settings.aws_region)
            
            # Initialize Pinecone client
            self.pinecone_client = Pinecone(
                api_key=self.settings.pinecone_api_key,
                environment=self.settings.pinecone_environment
            )
            self.pinecone_index = self.pinecone_client.Index(self.settings.pinecone_index_name)
            logger.info("Pinecone client initialized", index=self.settings.pinecone_index_name)
            
            # Test Bedrock connection
            self._test_bedrock_connection()
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS CLI or set environment variables.")
            raise
        except Exception as e:
            logger.error("Failed to initialize clients", error=str(e))
            raise
    
    def _test_bedrock_connection(self):
        """Test the Bedrock connection with a simple embedding request."""
        try:
            test_response = self.bedrock_client.invoke_model(
                modelId=self.settings.embed_model_id,
                body=json.dumps({"inputText": "test"}),
                contentType="application/json",
                accept="application/json",
            )
            logger.info("Bedrock connection test successful")
        except ClientError as e:
            if e.response['Error']['Code'] == 'AccessDeniedException':
                logger.error(
                    "Access denied to Bedrock. Ensure the model is enabled in your region",
                    model_id=self.settings.embed_model_id,
                    region=self.settings.aws_region
                )
            raise
    
    def _setup_text_splitter(self):
        """Initialize the text splitter for chunking documents."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
        )
        logger.info(
            "Text splitter configured",
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap
        )
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text using Bedrock Titan.
        
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
            
            logger.debug("Generated embedding", text_length=len(text), embedding_dim=len(embedding))
            return embedding
            
        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e), text_preview=text[:100])
            raise
    
    def create_document_id(self, source: str, chunk_index: int) -> str:
        """
        Create a unique document ID for a chunk.
        
        Args:
            source: Source file path
            chunk_index: Index of the chunk within the document
            
        Returns:
            Unique document ID
        """
        content = f"{source}:{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def process_document(self, file_path: pathlib.Path) -> List[Dict[str, Any]]:
        """
        Process a single document into chunks with embeddings.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of processed chunks with embeddings and metadata
        """
        logger.info("Processing document", file_path=str(file_path))
        
        try:
            # Load document
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
            
            if not documents:
                logger.warning("No content loaded from file", file_path=str(file_path))
                return []
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info("Document split into chunks", file_path=str(file_path), chunk_count=len(chunks))
            
            processed_chunks = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding = self.generate_embedding(chunk.page_content)
                    
                    # Create document ID
                    doc_id = self.create_document_id(str(file_path), i)
                    
                    # Prepare metadata
                    metadata = {
                        "text": chunk.page_content,
                        "source": str(file_path),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "file_name": file_path.name,
                        "file_size": file_path.stat().st_size,
                        "ingested_at": datetime.now(UTC).isoformat(),
                        "text_length": len(chunk.page_content)
                    }
                    
                    processed_chunks.append({
                        "id": doc_id,
                        "values": embedding,
                        "metadata": metadata
                    })
                    
                    logger.debug(
                        "Processed chunk",
                        doc_id=doc_id,
                        chunk_index=i,
                        text_length=len(chunk.page_content)
                    )
                    
                except Exception as e:
                    logger.error(
                        "Failed to process chunk",
                        file_path=str(file_path),
                        chunk_index=i,
                        error=str(e)
                    )
                    continue
            
            logger.info(
                "Document processing complete",
                file_path=str(file_path),
                processed_chunks=len(processed_chunks)
            )
            
            return processed_chunks
            
        except Exception as e:
            logger.error("Failed to process document", file_path=str(file_path), error=str(e))
            return []
    
    def ingest_to_pinecone(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Ingest processed chunks into Pinecone.
        
        Args:
            chunks: List of processed chunks with embeddings
            batch_size: Number of chunks to upload in each batch
        """
        if not chunks:
            logger.warning("No chunks to ingest")
            return
        
        logger.info("Starting Pinecone ingestion", total_chunks=len(chunks))
        
        try:
            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Prepare vectors for upsert
                vectors = [(chunk["id"], chunk["values"], chunk["metadata"]) for chunk in batch]
                
                # Upsert to Pinecone
                self.pinecone_index.upsert(vectors=vectors)
                
                logger.info(
                    "Batch uploaded to Pinecone",
                    batch_start=i,
                    batch_size=len(batch),
                    total_uploaded=min(i + batch_size, len(chunks))
                )
            
            logger.info("Pinecone ingestion complete", total_chunks=len(chunks))
            
        except Exception as e:
            logger.error("Failed to ingest to Pinecone", error=str(e))
            raise
    
    def ingest_directory(self, directory_path: str):
        """
        Ingest all text files from a directory.
        
        Args:
            directory_path: Path to the directory containing documents
        """
        directory = pathlib.Path(directory_path)
        
        if not directory.exists():
            logger.error("Directory does not exist", directory=str(directory))
            return
        
        # Find all text files
        text_files = list(directory.rglob("*.txt"))
        
        if not text_files:
            logger.warning("No .txt files found in directory", directory=str(directory))
            return
        
        logger.info("Starting directory ingestion", directory=str(directory), file_count=len(text_files))
        
        all_chunks = []
        successful_files = 0
        
        for file_path in text_files:
            try:
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)
                successful_files += 1
                
            except Exception as e:
                logger.error("Failed to process file", file_path=str(file_path), error=str(e))
                continue
        
        if all_chunks:
            self.ingest_to_pinecone(all_chunks)
            
            # Print summary
            try:
                index_stats = self.pinecone_index.describe_index_stats()
                index_stats_dict = self._serialize_index_stats(index_stats)
            except Exception as e:
                logger.warning("Failed to get index stats", error=str(e))
                index_stats_dict = {"error": "Unable to retrieve stats"}
            
            try:
                logger.info(
                    "Ingestion complete",
                    files_processed=successful_files,
                    total_files=len(text_files),
                    chunks_ingested=len(all_chunks),
                    index_stats=index_stats_dict
                )
            except Exception as e:
                import traceback
                logger.error("Failed to log completion message", error=str(e), traceback=traceback.format_exc())
                # Fallback logging without index_stats
                logger.info(
                    "Ingestion complete (basic)",
                    files_processed=successful_files,
                    total_files=len(text_files),
                    chunks_ingested=len(all_chunks)
                )
        else:
            logger.warning("No chunks were successfully processed")
    
    def ingest_s3_bucket(self, bucket_name: str, prefix: str = ""):
        """
        Ingest documents from an S3 bucket.
        
        Args:
            bucket_name: Name of the S3 bucket
            prefix: Optional prefix to filter objects
        """
        try:
            s3_client = boto3.client('s3', region_name=self.settings.aws_region)
            
            # List objects in bucket
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            
            if 'Contents' not in response:
                logger.warning("No objects found in S3 bucket", bucket=bucket_name, prefix=prefix)
                return
            
            objects = response['Contents']
            text_objects = [obj for obj in objects if obj['Key'].endswith('.txt')]
            
            logger.info(
                "Found text files in S3",
                bucket=bucket_name,
                prefix=prefix,
                file_count=len(text_objects)
            )
            
            all_chunks = []
            successful_files = 0
            
            for obj in text_objects:
                try:
                    # Download file content
                    response = s3_client.get_object(Bucket=bucket_name, Key=obj['Key'])
                    content = response['Body'].read().decode('utf-8')
                    
                    # Create a document object
                    doc = Document(page_content=content, metadata={"source": f"s3://{bucket_name}/{obj['Key']}"})
                    
                    # Split into chunks
                    chunks = self.text_splitter.split_documents([doc])
                    
                    processed_chunks = []
                    for i, chunk in enumerate(chunks):
                        embedding = self.generate_embedding(chunk.page_content)
                        doc_id = self.create_document_id(f"s3://{bucket_name}/{obj['Key']}", i)
                        
                        metadata = {
                            "text": chunk.page_content,
                            "source": f"s3://{bucket_name}/{obj['Key']}",
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "file_name": obj['Key'].split('/')[-1],
                            "file_size": obj['Size'],
                            "ingested_at": datetime.now(UTC).isoformat(),
                            "text_length": len(chunk.page_content)
                        }
                        
                        processed_chunks.append({
                            "id": doc_id,
                            "values": embedding,
                            "metadata": metadata
                        })
                    
                    all_chunks.extend(processed_chunks)
                    successful_files += 1
                    
                    logger.info(
                        "Processed S3 file",
                        file_key=obj['Key'],
                        chunks=len(processed_chunks)
                    )
                    
                except Exception as e:
                    logger.error("Failed to process S3 file", file_key=obj['Key'], error=str(e))
                    continue
            
            if all_chunks:
                self.ingest_to_pinecone(all_chunks)
                
                try:
                    index_stats = self.pinecone_index.describe_index_stats()
                    index_stats_dict = self._serialize_index_stats(index_stats)
                except Exception as e:
                    logger.warning("Failed to get index stats", error=str(e))
                    index_stats_dict = {"error": "Unable to retrieve stats"}
                    
                logger.info(
                    "S3 ingestion complete",
                    files_processed=successful_files,
                    total_files=len(text_objects),
                    chunks_ingested=len(all_chunks),
                    index_stats=index_stats_dict
                )
            else:
                logger.warning("No chunks were successfully processed from S3")
                
        except Exception as e:
            logger.error("Failed to ingest from S3", bucket=bucket_name, error=str(e))
            raise
    
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


def main():
    """Main entry point for the ingestion script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG pipeline")
    parser.add_argument(
        "--source-type",
        choices=["local", "s3"],
        default="local",
        help="Source type for documents"
    )
    parser.add_argument(
        "--path",
        default="./data",
        help="Local directory path or S3 bucket name"
    )
    parser.add_argument(
        "--s3-prefix",
        default="",
        help="S3 prefix for filtering objects (only for S3 source)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    try:
        ingester = DocumentIngester()
        
        if args.source_type == "local":
            ingester.ingest_directory(args.path)
        elif args.source_type == "s3":
            ingester.ingest_s3_bucket(args.path, args.s3_prefix)
        
        logger.info("Ingestion completed successfully")
        
    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main() 