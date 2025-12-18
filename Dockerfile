FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set HOME to writable directory (important for Streamlit)
ENV HOME=/app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv pip install --system --no-cache .

# Copy application files
COPY app.py ./

# Create .streamlit directory and config
RUN mkdir -p /app/.streamlit && \
    echo "[server]\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\n[browser]\ngatherUsageStats = false" > /app/.streamlit/config.toml

# Set proper permissions for OpenShift (runs as random UID with GID 0)
RUN chmod -R g+rwX /app

# Expose port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]