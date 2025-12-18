FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies with uv
RUN uv pip install --system --no-cache .

# Copy application files
COPY app.py ./

# OpenShift runs as arbitrary UID - set group permissions
RUN chmod -R g+rwX /app

# Expose port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]