# =====================================================================
# STAGE 1: Build TypeScript + React Frontend (Vite)
# =====================================================================
FROM node:22-alpine AS frontend-builder

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm ci || npm install

COPY frontend/ ./
RUN npm run build

# =====================================================================
# STAGE 2: Python ML + FastAPI Backend + Static SPA Bundle
# =====================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

# libgomp1 este necesar pentru rularea XGBoost pe arhitectură Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Copiem bundle-ul TypeScript compilat în stage-ul anterior
COPY --from=frontend-builder /frontend/dist /app/frontend/dist

# Asigură generarea seturilor de date și a modelului antrenat în imagine
RUN python train.py

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8501"]
