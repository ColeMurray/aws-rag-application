#!/usr/bin/env python3
"""
Quick start script for AWS RAG Application.

This script helps users get started quickly by checking prerequisites
and guiding them through the setup process.
"""
import os
import sys
import subprocess
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass


def print_banner():
    """Print welcome banner."""
    print("🚀 AWS RAG Application - Quick Start")
    print("=" * 50)
    print("This script will help you set up the RAG pipeline quickly.")
    print()


def check_python_version():
    """Check if Python version is compatible."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10 or higher is required.")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n🔍 Checking dependencies...")
    
    # Map package names to their import names
    package_imports = {
        "boto3": "boto3",
        "pinecone": "pinecone",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "langchain": "langchain",
        "pydantic": "pydantic",
        "structlog": "structlog"
    }
    
    missing_packages = []
    
    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Not installed")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True


def check_environment_variables():
    """Check if required environment variables are set."""
    print("\n🔍 Checking environment variables...")
    
    required_vars = [
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT"
    ]
    
    optional_vars = [
        "AWS_REGION",
        "PINECONE_INDEX_NAME",
        "BEDROCK_EMBED_MODEL_ID",
        "BEDROCK_LLM_MODEL_ID"
    ]
    
    missing_required = []
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}")
        else:
            print(f"❌ {var} - Not set")
            missing_required.append(var)
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var} (optional)")
        else:
            print(f"⚪ {var} - Using default")
    
    if missing_required:
        print(f"\n⚠️  Missing required variables: {', '.join(missing_required)}")
        print("Create a .env file or set these environment variables.")
        return False
    
    return True


def check_aws_credentials():
    """Check if AWS credentials are configured."""
    print("\n🔍 Checking AWS credentials...")
    
    try:
        import boto3
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials:
            print("✅ AWS credentials found")
            region = session.region_name or os.getenv('AWS_REGION', 'us-east-1')
            print(f"✅ AWS region: {region}")
            return True
        else:
            print("❌ AWS credentials not found")
            print("Configure with: aws configure")
            return False
            
    except Exception as e:
        print(f"❌ AWS check failed: {e}")
        return False


def check_sample_data():
    """Check if sample data exists."""
    print("\n🔍 Checking sample data...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found")
        return False
    
    txt_files = list(data_dir.glob("*.txt"))
    if txt_files:
        print(f"✅ Found {len(txt_files)} sample documents")
        for file in txt_files:
            print(f"   - {file.name}")
        return True
    else:
        print("❌ No sample documents found in data/ directory")
        return False


def run_ingestion():
    """Run the ingestion script."""
    print("\n🔄 Running document ingestion...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "src.ingest", 
            "--source-type", "local",
            "--path", "./data",
            "--log-level", "INFO"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Document ingestion completed successfully")
            print("   Check the logs above for details")
            return True
        else:
            print("❌ Document ingestion failed")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Document ingestion timed out")
        return False
    except Exception as e:
        print(f"❌ Failed to run ingestion: {e}")
        return False


def start_api():
    """Start the API server."""
    print("\n🚀 Starting API server...")
    print("The server will start on http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "src.app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")


def show_next_steps():
    """Show next steps to the user."""
    print("\n🎉 Setup completed! Next steps:")
    print()
    print("1. 📖 API Documentation: http://localhost:8000/docs")
    print("2. 🔍 Health Check: http://localhost:8000/health")
    print("3. 📊 Index Stats: http://localhost:8000/stats")
    print()
    print("📝 Test the API with curl:")
    print('curl -X POST "http://localhost:8000/query" \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"query": "What is the remote work policy?"}\'')
    print()
    print("🧪 Or run the test suite:")
    print("python scripts/test_api.py")
    print()
    print("📚 For more information, see README.md")


def main():
    """Main function."""
    print_banner()
    
    # Pre-flight checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment Variables", check_environment_variables),
        ("AWS Credentials", check_aws_credentials),
        ("Sample Data", check_sample_data),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        if not check_func():
            all_passed = False
            break
    
    if not all_passed:
        print("\n❌ Pre-flight checks failed. Please fix the issues above and try again.")
        print("\n📚 See README.md for detailed setup instructions.")
        sys.exit(1)
    
    print("\n✅ All pre-flight checks passed!")
    
    # Ask user if they want to run ingestion
    print("\n" + "=" * 50)
    response = input("🤔 Run document ingestion now? (y/N): ").lower().strip()
    
    if response in ['y', 'yes']:
        if not run_ingestion():
            print("\n❌ Ingestion failed. You can run it manually later:")
            print("python -m src.ingest --source-type local --path ./data")
    else:
        print("⏩ Skipping ingestion. Run manually when ready:")
        print("python -m src.ingest --source-type local --path ./data")
    
    # Ask user if they want to start the API
    print("\n" + "=" * 50)
    response = input("🤔 Start the API server now? (Y/n): ").lower().strip()
    
    if response not in ['n', 'no']:
        show_next_steps()
        print("\n" + "=" * 50)
        print("🚀 Starting server in 3 seconds... (Press Ctrl+C to cancel)")
        
        try:
            import time
            time.sleep(3)
            start_api()
        except KeyboardInterrupt:
            print("\n⏹️  Cancelled by user")
    else:
        print("\n⏩ Server not started. Start manually when ready:")
        print("uvicorn src.app:app --reload --port 8000")
        show_next_steps()


if __name__ == "__main__":
    main() 