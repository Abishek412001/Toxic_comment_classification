# Enterprise DEPLOYMENT.md

## Production Deployment Checklist & Step-by-Step Guide

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- 4 GB RAM minimum

### Cloud Deployments
- **Streamlit Community Cloud**: Entry point `dashboard/app.py`.
- **Docker Production Container**: `docker-compose -f deployment/docker-compose.yml up -d`.
