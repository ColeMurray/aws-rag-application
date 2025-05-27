# Docker Deployment for AWS RAG Application

This directory contains Docker configuration files for deploying the AWS RAG (Retrieval-Augmented Generation) application.

## Prerequisites

- Docker (version 20.10 or later)
- Docker Compose (version 2.0 or later)
- AWS credentials configured (via AWS CLI, environment variables, or IAM roles)
- Pinecone API key and index

## Quick Start

1. **Create environment file:**
   ```bash
   # From the project root
   cp env.example .env
   # Edit .env with your actual credentials
   ```

2. **Start the application:**
   ```bash
   cd deployment/docker
   docker-compose up -d
   ```

3. **Check the application:**
   - Health check: http://localhost:8000/health
   - API documentation: http://localhost:8000/docs
   - Root endpoint: http://localhost:8000/

## Configuration

### Environment Variables

The application requires the following environment variables:

#### Required
- `PINECONE_API_KEY`: Your Pinecone API key
- `AWS_REGION`: AWS region (default: us-east-1)

#### Optional (with defaults)
- `PINECONE_ENVIRONMENT`: Pinecone environment (default: us-east-1-aws)
- `PINECONE_INDEX_NAME`: Pinecone index name (default: rag-documents)
- `BEDROCK_EMBED_MODEL_ID`: Embedding model (default: amazon.titan-embed-text-v2:0)
- `BEDROCK_LLM_MODEL_ID`: LLM model (default: us.anthropic.claude-sonnet-4-20250514-v1:0)
- `LOG_LEVEL`: Logging level (default: INFO)

### AWS Credentials

You can provide AWS credentials in several ways:

1. **Environment variables** (in .env file):
   ```
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   ```

2. **AWS CLI configuration** (recommended):
   ```bash
   aws configure
   ```

3. **IAM roles** (for EC2/ECS deployment)

4. **Instance profiles** (for EC2 deployment)

## Files

- `Dockerfile`: Multi-stage Docker build configuration
- `docker-compose.yml`: Service orchestration configuration
- `README.md`: This documentation

## Docker Commands

### Build and Start
```bash
# Build and start services
docker-compose up -d

# Build without cache
docker-compose build --no-cache

# Start with logs
docker-compose up
```

### Management
```bash
# Stop services
docker-compose down

# Restart services
docker-compose restart

# View logs
docker-compose logs -f rag-api

# Check status
docker-compose ps
```

### Development
```bash
# Start with auto-reload (for development)
docker-compose up -d

# Execute commands in container
docker-compose exec rag-api bash

# View real-time logs
docker-compose logs -f rag-api
```

## Testing the Deployment

After starting the services, you can test the deployment manually:

1. **Check service health:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Test a sample query:**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What are AWS security best practices?"}'
   ```

3. **View API documentation:**
   ```bash
   open http://localhost:8000/docs
   ```

4. **Check container status:**
   ```bash
   docker-compose ps
   ```

## Troubleshooting

### Common Issues

1. **Port 8000 already in use**
   ```bash
   # Find and kill the process using port 8000
   lsof -i :8000
   kill -9 <PID>
   ```

2. **Docker build fails**
   ```bash
   # Clean Docker cache and rebuild
   docker system prune -f
   docker-compose build --no-cache
   ```

3. **Service fails to start**
   ```bash
   # Check logs for errors
   docker-compose logs rag-api
   
   # Check container status
   docker-compose ps
   ```

4. **Health check fails**
   ```bash
   # Check if the service is responding
   curl -v http://localhost:8000/health
   
   # Check application logs
   docker-compose logs rag-api
   ```

5. **AWS/Pinecone connection issues**
   - Verify your credentials in the .env file
   - Check AWS region and Pinecone environment settings
   - Ensure Bedrock models are enabled in your AWS account
   - Verify Pinecone index exists and is accessible

### Debug Mode

To run in debug mode with more verbose logging:

1. Update docker-compose.yml:
   ```yaml
   environment:
     - LOG_LEVEL=DEBUG
   ```

2. Restart the service:
   ```bash
   docker-compose restart
   ```

### Performance Tuning

For production deployment, consider:

1. **Resource limits** in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 2G
         cpus: '1.0'
   ```

2. **Health check intervals**:
   ```yaml
   healthcheck:
     interval: 60s
     timeout: 30s
     retries: 3
   ```

3. **Logging configuration**:
   ```yaml
   logging:
     driver: "json-file"
     options:
       max-size: "10m"
       max-file: "3"
   ```

## Security Considerations

1. **Never commit .env files** with real credentials
2. **Use secrets management** for production (Docker Secrets, AWS Secrets Manager)
3. **Run as non-root user** (already configured in Dockerfile)
4. **Use specific image tags** instead of `latest` for production
5. **Regularly update base images** for security patches

## Production Deployment

For production deployment:

1. Use a reverse proxy (nginx, traefik)
2. Enable HTTPS/TLS
3. Configure proper logging and monitoring
4. Use container orchestration (Docker Swarm, Kubernetes)
5. Implement proper backup strategies
6. Set up CI/CD pipelines

## Support

If you encounter issues:

1. Check the logs: `docker-compose logs rag-api`
2. Verify your environment configuration
3. Ensure all prerequisites are met
4. Test endpoints manually using curl commands above 