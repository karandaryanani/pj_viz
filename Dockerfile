FROM quay.io/centos/centos:stream9

# Install Python 3.12
RUN dnf install -y python3.12 python3.12-pip && \
    dnf clean all

# Set working directory
WORKDIR /app
ENV HOME=/app

# Install uv
RUN pip3.12 install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv pip install --system --no-cache .

# Copy application files
COPY app.py ./

# Create directories
RUN mkdir -p /app/.streamlit /app/data && \
    echo "[server]\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\n[browser]\ngatherUsageStats = false" > /app/.streamlit/config.toml

# Set permissions
RUN chmod -R g+rwX /app

# Expose port
EXPOSE 8501

# Run app
CMD ["python3.12", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]