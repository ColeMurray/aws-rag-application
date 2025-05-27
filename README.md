# AWS RAG Application

A **Retrieval-Augmented Generation (RAG) pipeline** demonstration using AWS Bedrock and Pinecone. This application shows how to build an intelligent document query system that can answer questions about your documents using AI.

## 🏗️ Project Structure

```
aws-rag-application/
├── README.md                    # This file - setup and usage guide
├── requirements.txt             # Python dependencies
├── env.example                  # Environment configuration template
├── 
├── src/                         # Core application code
│   ├── __init__.py             # Package initialization
│   ├── app.py                  # FastAPI application (REST API)
│   ├── config.py               # Configuration management
│   └── ingest.py               # Document ingestion pipeline
│   
├── scripts/                     # Utility scripts
│   ├── quickstart.py           # Interactive setup wizard
│   └── test_api.py             # API testing suite
│   
├── data/                        # Sample documents
│   ├── sample_company_policy.txt
│   └── aws_best_practices.txt
│   
└── deployment/                  # Deployment configurations
    └── docker/
        ├── Dockerfile
        ├── docker-compose.yml
        └── README.md           # Docker deployment guide
```

## 🚀 Quick Start

The fastest way to get started is using our interactive setup script:

```bash
# Clone the repository
git clone <repository-url>
cd aws-rag-application

# Install dependencies
pip install -r requirements.txt

# Run the interactive setup
python scripts/quickstart.py
```

The quickstart script will:
- ✅ Check Python version and dependencies
- ✅ Verify AWS credentials and environment variables
- ✅ Test connections to AWS Bedrock and Pinecone
- ✅ Run document ingestion
- ✅ Start the API server

## 📋 Prerequisites

### Required Services
- **AWS Account** with Bedrock enabled (us-east-1 or us-west-2 recommended)
- **Pinecone Account** (free tier available)
- **Python 3.10+**

### AWS Setup
1. **Enable Bedrock Models** in your AWS region:
   - Titan Text Embeddings V2
   - Claude 3 Sonnet
2. **Configure AWS credentials** using one of these methods:
   ```bash
   # Option 1: AWS CLI
   aws configure
   
   # Option 2: Environment variables
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   export AWS_REGION=us-east-1
   ```

### Pinecone Setup
1. **Create a Pinecone account** at [pinecone.io](https://pinecone.io)
2. **Create an index** with these settings:
   - **Dimension**: 1024 (for Titan Embeddings V2)
   - **Metric**: cosine
   - **Pod Type**: Starter (free tier)
3. **Get your API key** from the Pinecone console

## ⚙️ Configuration

Create a `.env` file in the project root:

```bash
# Required - Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_environment  # e.g., "us-east-1-aws"
PINECONE_INDEX_NAME=rag-demo-index

# Optional - AWS Configuration (if not using AWS CLI)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Optional - Model Configuration
BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_LLM_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Optional - Processing Parameters
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K=6
SIMILARITY_THRESHOLD=0.7
```

## 🔄 Manual Setup (Step by Step)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy the configuration example above to a `.env` file and update with your values.

### 3. Ingest Documents
```bash
# Ingest from local data directory
python src/ingest.py --source-type local --path ./data

# Or ingest from S3 bucket
python src/ingest.py --source-type s3 --path your-bucket-name --s3-prefix documents/
```

### 4. Start the API Server
```bash
# Start the FastAPI server
python src/app.py

# Or use uvicorn directly
uvicorn src.app:app --reload --port 8000
```

### 5. Test the API
```bash
# Run the test suite
python scripts/test_api.py

# Or test manually
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the remote work policy?"}'
```

## 🔗 API Endpoints

### Query Documents
**POST** `/query`
```json
{
  "query": "What is the remote work policy?",
  "max_chunks": 5,
  "similarity_threshold": 0.7,
  "include_sources": true
}
```

**Response:**
```json
{
  "answer": "Based on the company policy...",
  "query": "What is the remote work policy?",
  "sources": [
    {
      "source": "./data/sample_company_policy.txt",
      "chunk_index": 2,
      "similarity_score": 0.89,
      "text_preview": "Remote work is permitted for eligible employees..."
    }
  ],
  "processing_time_ms": 1250.5,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### Health Check
**GET** `/health`
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "version": "1.0.0",
  "services": {
    "bedrock": "healthy",
    "pinecone": "healthy"
  }
}
```

### Index Statistics
**GET** `/stats`
```json
{
  "index_stats": {
    "total_vector_count": 156,
    "dimension": 1024
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🐳 Docker Deployment

### Development with Docker Compose
```bash
# Build and start all services
docker-compose -f deployment/docker/docker-compose.yml up --build

# Run in background
docker-compose -f deployment/docker/docker-compose.yml up -d
```

### Production Docker
```bash
# Build the image
docker build -f deployment/docker/Dockerfile -t aws-rag-app .

# Run the container
docker run -p 8000:8000 --env-file .env aws-rag-app
```

## ☁️ AWS Deployment Options

### Option 1: ECS Fargate (Recommended)
```bash
# Using AWS Copilot
copilot app init aws-rag-app
copilot env init --name production
copilot svc init --name api --svc-type "Load Balanced Web Service"
copilot svc deploy --name api --env production
```

### Option 2: Lambda + API Gateway
- Package the application as a Lambda function
- Use Mangum for ASGI to AWS Lambda adapter
- Deploy with AWS SAM or CDK

### Option 3: EC2 with Docker
- Launch EC2 instance with Docker
- Use the provided Dockerfile
- Set up Application Load Balancer

## 💰 Cost Estimation

### Development Environment (~$5-10/month)
| Service | Usage | Cost |
|---------|-------|------|
| **AWS Bedrock Titan Embeddings** | 50K requests | ~$0.50 |
| **AWS Bedrock Claude 3 Sonnet** | 100K input + 10K output tokens | ~$3.00 |
| **Pinecone Starter** | 1 pod, <100K vectors | **Free** |
| **AWS Lambda** | 100K requests | **Free tier** |
| **S3 Storage** | 1GB | **Free tier** |

### Production Environment (scales with usage)
- **Bedrock**: $0.0001 per 1K input tokens, $0.0005 per 1K output tokens
- **Pinecone**: $70/month per pod for production workloads
- **ECS Fargate**: $0.04048 per vCPU per hour, $0.004445 per GB per hour

## 🔧 Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PINECONE_API_KEY` | ✅ | - | Pinecone API key |
| `PINECONE_ENVIRONMENT` | ✅ | - | Pinecone environment (e.g., "us-east-1-aws") |
| `PINECONE_INDEX_NAME` | ❌ | "rag-demo-index" | Name of the Pinecone index |
| `AWS_REGION` | ❌ | "us-east-1" | AWS region for Bedrock |
| `BEDROCK_EMBED_MODEL_ID` | ❌ | "amazon.titan-embed-text-v2:0" | Embedding model |
| `BEDROCK_LLM_MODEL_ID` | ❌ | "anthropic.claude-3-sonnet-20240229-v1:0" | LLM model |
| `CHUNK_SIZE` | ❌ | 800 | Characters per text chunk |
| `CHUNK_OVERLAP` | ❌ | 100 | Overlap between chunks |
| `TOP_K` | ❌ | 6 | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | ❌ | 0.7 | Minimum similarity score |

### Model Configuration

#### Supported Embedding Models
- `amazon.titan-embed-text-v2:0` (1024 dimensions) - **Recommended**
- `amazon.titan-embed-text-v1` (1024 dimensions)

#### Supported LLM Models
- `anthropic.claude-3-sonnet-20240229-v1:0` - **Recommended**
- `anthropic.claude-3-haiku-20240307-v1:0` (faster, lower cost)
- `anthropic.claude-3-opus-20240229-v1:0` (highest quality)

## 🔍 Usage Examples

### Python Client
```python
import requests

# Query the API
response = requests.post("http://localhost:8000/query", json={
    "query": "What are the AWS security best practices?",
    "max_chunks": 3,
    "include_sources": True
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])}")
```

### cURL Examples
```bash
# Basic query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the expense reporting process?"}'

# Query with parameters
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "AWS monitoring recommendations",
    "max_chunks": 3,
    "similarity_threshold": 0.8,
    "include_sources": true
  }'

# Health check
curl "http://localhost:8000/health"

# Index statistics
curl "http://localhost:8000/stats"
```

## 🧪 Testing

### Run the Test Suite
```bash
# Comprehensive API tests
python scripts/test_api.py

# Test with different parameters
python scripts/test_api.py --verbose
```

### Manual Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test with sample queries
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "remote work equipment policy"}'
```

## 🛠️ Troubleshooting

### Common Issues

#### "No module named 'src'"
Make sure you're running commands from the project root directory:
```bash
# From project root
python src/ingest.py
python src/app.py
```

#### AWS Credentials Not Found
```bash
# Configure AWS CLI
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
```

#### Bedrock Access Denied
1. Ensure Bedrock is enabled in your region
2. Enable the specific models you want to use
3. Check IAM permissions for `bedrock:InvokeModel`

#### Pinecone Connection Issues
1. Verify API key and environment in `.env` file
2. Check index name matches your Pinecone console
3. Ensure index dimension matches embedding model (1024 for Titan V2)

#### No Documents Found
```bash
# Check data directory exists
ls -la data/

# Verify files are .txt format
find data/ -name "*.txt"

# Run ingestion with verbose logging
python src/ingest.py --source-type local --path ./data --log-level DEBUG
```

### Debug Mode
Enable debug logging:
```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or run with debug flag
python src/app.py --debug
```

### Health Checks
```bash
# Check all services
curl http://localhost:8000/health

# Check index stats
curl http://localhost:8000/stats
```

## 🔒 Security Considerations

### For Production Deployment
1. **Environment Variables**: Store secrets in AWS Secrets Manager or similar
2. **IAM Roles**: Use least-privilege IAM roles instead of access keys
3. **CORS**: Configure appropriate CORS settings for your domain
4. **Rate Limiting**: Implement API rate limiting
5. **Input Validation**: The API includes comprehensive input validation
6. **HTTPS**: Always use HTTPS in production
7. **Network Security**: Use VPC and security groups appropriately

### IAM Policy Example
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/amazon.titan-*",
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": [
        "arn:aws:s3:::your-documents-bucket/*"
      ]
    }
  ]
}
```

## 📚 Additional Resources

### Documentation
- [Docker Deployment Guide](deployment/docker/README.md)
- [Environment Configuration](.env.template)

### External Resources
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://docs.langchain.com/)

## 🤝 Contributing

This project serves as a consulting asset and reference implementation. For improvements or customizations:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is provided as-is for educational and consulting purposes. Please review and comply with the licenses of all dependencies.

---

**🎯 Ready to get started?** Run `python scripts/quickstart.py` and follow the interactive setup!