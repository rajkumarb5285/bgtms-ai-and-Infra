# BGT-M2: Production Containerization & Docker Orchestration

**Milestone:** M2 | **Track:** Infrastructure & Deployment | **Status:** `COMPLETED`

## 1. Overview
The BGTMS AI & Serving Engine is containerized using multi-stage Docker builds and Docker Compose orchestration. It decouples the core inference/telemetry backend from the frontend presentation layer and enables instant edge deployment on Linux servers, AWS EC2 instances, and NVIDIA Jetson edge systems.

---

## 2. Docker Architecture

### 2.1 Multi-Stage Dockerfile (`Dockerfile`)
- **Base Image:** `python:3.11-slim` (minimal security attack surface, glibc compatibility for PyTorch and ONNX Runtime).
- **System Dependencies:**
  - `libgl1` and `libglib2.0-0`: Native graphics and multimedia shared libraries required by OpenCV.
  - `curl`: Used by the Docker daemon healthcheck probe.
- **Python ML Dependencies:** Installed via pinned `requirements.txt` with wheel caching (`--no-cache-dir`).
- **Healthcheck Probe:** Executes `curl -f http://localhost:8000/api/v1/health` every 20 seconds.
- **Port Exposure:** Port `8000` for REST API and WebSocket streams.

### 2.2 Orchestration (`docker-compose.yml`)
The composition defines two primary services connected via a private bridge network (`bgtms-net`):

```yaml
version: '3.8'

services:
  bgtms-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: bgtms-engine
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - AWS_DEFAULT_REGION=ap-south-1
      - S3_BUCKET_NAME=bgtms-ai-datasets
      - MODEL_DIR=/app/bgtms/models
    volumes:
      - ./data:/app/data
      - ./configs:/app/configs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 20s
      timeout: 5s
      retries: 3
    networks:
      - bgtms-net

  bgtms-dashboard:
    image: nginx:alpine
    container_name: bgtms-web-dashboard
    restart: unless-stopped
    ports:
      - "8080:80"
    volumes:
      - ./dashboard:/usr/share/nginx/html:ro
      - ./docker/nginx.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      - bgtms-api
    networks:
      - bgtms-net

networks:
  bgtms-net:
    driver: bridge
```

### 2.3 Reverse Proxy & WebSocket Routing (`docker/nginx.conf`)
- Routes root HTTP traffic to static dashboard UI.
- Upgrades `/ws/` connections using HTTP/1.1 WebSockets with `Upgrade` and `Connection: Upgrade` headers.
- Proxies `/api/` REST calls transparently to the backend service.

---

## 3. Deployment Commands
```bash
# Build and start all services in detached mode
docker compose up -d --build

# View real-time container logs
docker compose logs -f

# Inspect health check status
docker compose ps

# Shutdown services
docker compose down
```
