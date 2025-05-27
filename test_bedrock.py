#!/usr/bin/env python3
"""Test Bedrock model access."""

import boto3
import json
from dotenv import load_dotenv

load_dotenv()

def test_embedding_model():
    """Test the embedding model."""
    try:
        client = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        response = client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({"inputText": "test"}),
            contentType="application/json",
            accept="application/json",
        )
        
        print("✅ Embedding model works!")
        result = json.loads(response["body"].read())
        print(f"   Embedding dimension: {len(result['embedding'])}")
        
    except Exception as e:
        print(f"❌ Embedding model error: {e}")
        if "AccessDeniedException" in str(e):
            print("   → Model needs to be enabled in Bedrock console")
        elif "ValidationException" in str(e):
            print("   → Model ID is invalid or not available")

def test_llm_model():
    """Test the LLM model."""
    try:
        client = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        response = client.invoke_model(
            modelId="us.anthropic.claude-sonnet-4-20250514-v1:0",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "temperature": 0.5,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello there"}],
                    }
                ]})
        )
        
        print("✅ LLM model works!")
        result = json.loads(response["body"].read())
        print(f"   Response: {result}")
        
    except Exception as e:
        print(f"❌ LLM model error: {e}")
        if "AccessDeniedException" in str(e):
            print("   → Model needs to be enabled in Bedrock console")
        elif "ValidationException" in str(e):
            print("   → Model ID is invalid or not available")

if __name__ == "__main__":
    print("Testing Bedrock model access...")
    print()
    test_embedding_model()
    print()
    test_llm_model()
    print()
    print("If you see 'AccessDeniedException', you need to:")
    print("1. Go to AWS Bedrock console")
    print("2. Navigate to 'Model access'")
    print("3. Enable the required models")
    print("4. Wait for approval (can take a few minutes)") 