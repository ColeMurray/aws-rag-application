#!/usr/bin/env python3
"""
Simple test script for the RAG API.

This script tests the main functionality of the RAG pipeline
to ensure everything is working correctly.
"""
import json
import time
import requests
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_QUERIES = [
    "What is the remote work policy for equipment?",
    "What are the AWS security best practices for IAM?",
    "How often should employees submit expense reports?",
    "What monitoring tools are recommended for AWS?",
]


def test_health_check() -> bool:
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        
        health_data = response.json()
        print(f"✅ Health Status: {health_data['status']}")
        print(f"   Services: {health_data['services']}")
        
        return health_data['status'] in ['healthy', 'degraded']
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_root_endpoint() -> bool:
    """Test the root endpoint."""
    print("\n🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=10)
        response.raise_for_status()
        
        root_data = response.json()
        print(f"✅ API: {root_data['message']}")
        print(f"   Version: {root_data['version']}")
        
        return True
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False


def test_stats_endpoint() -> bool:
    """Test the stats endpoint."""
    print("\n🔍 Testing stats endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        response.raise_for_status()
        
        stats_data = response.json()
        print(f"✅ Index stats retrieved")
        
        if 'index_stats' in stats_data:
            index_stats = stats_data['index_stats']
            total_vectors = index_stats.get('total_vector_count', 0)
            print(f"   Total vectors in index: {total_vectors}")
            
            if total_vectors == 0:
                print("⚠️  Warning: No vectors found in index. Run ingestion first.")
        
        return True
    except Exception as e:
        print(f"❌ Stats endpoint failed: {e}")
        return False


def test_query_endpoint(query: str) -> Dict[str, Any]:
    """Test a single query."""
    print(f"\n🔍 Testing query: '{query}'")
    try:
        payload = {
            "query": query,
            "max_chunks": 5,
            "similarity_threshold": 0.5,
            "include_sources": True
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        response_time = time.time() - start_time
        
        result = response.json()
        
        print(f"✅ Query successful ({response_time:.2f}s)")
        print(f"   Answer: {result['answer'][:100]}...")
        print(f"   Sources found: {len(result.get('sources', []))}")
        print(f"   Processing time: {result.get('processing_time_ms', 0):.1f}ms")
        
        return result
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return {}


def test_invalid_query() -> bool:
    """Test invalid query handling."""
    print("\n🔍 Testing invalid query handling...")
    try:
        # Test empty query
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": ""},
            timeout=10
        )
        
        if response.status_code == 400:
            print("✅ Empty query properly rejected")
            return True
        else:
            print(f"❌ Expected 400 status, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Invalid query test failed: {e}")
        return False


def run_all_tests():
    """Run all API tests."""
    print("🚀 Starting RAG API Tests")
    print("=" * 50)
    
    test_results = []
    
    # Basic endpoint tests
    test_results.append(("Health Check", test_health_check()))
    test_results.append(("Root Endpoint", test_root_endpoint()))
    test_results.append(("Stats Endpoint", test_stats_endpoint()))
    test_results.append(("Invalid Query", test_invalid_query()))
    
    # Query tests
    for i, query in enumerate(TEST_QUERIES):
        result = test_query_endpoint(query)
        test_results.append((f"Query {i+1}", bool(result)))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, passed_test in test_results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} {test_name}")
        if passed_test:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The RAG API is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


def main():
    """Main test function."""
    print("RAG API Test Suite")
    print("Make sure the API is running on http://localhost:8000")
    print("Run 'uvicorn src.app:app --reload' or 'python scripts/quickstart.py' first\n")
    
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main() 