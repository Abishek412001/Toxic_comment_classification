# Enterprise Multi-Platform Deployment Guide (Step 137)

## 1. Local Docker Container Deployment
```bash
# Build Docker image
docker build -t toxic-comment-app:latest -f deployment/Dockerfile .

# Run Docker container
docker run -d -p 8501:8501 --name toxic_app toxic-comment-app:latest

# Or launch via Docker Compose
docker-compose -f deployment/docker-compose.yml up -d
```

---

## 2. Streamlit Community Cloud Deployment
1. Push repository to GitHub.
2. Log into Streamlit Cloud (`share.streamlit.io`).
3. Connect repository branch `main` and set entry point file to `dashboard/app.py`.
4. Configure Secret Environment Variables (`APP_ENV=production`).

---

## 3. Render Web Service Deployment
1. Connect GitHub repo to Render Dashboard.
2. Select **Docker Runtime**.
3. Specify Dockerfile path: `deployment/Dockerfile`.
4. Expose port `8501`.
