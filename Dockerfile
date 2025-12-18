FROM quay.io/fedora/python-312:latest

WORKDIR /app
ENV HOME=/app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache .

COPY app.py ./

RUN mkdir -p /app/.streamlit /app/data && \
    echo "[server]\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\n[browser]\ngatherUsageStats = false" > /app/.streamlit/config.toml

RUN chmod -R g+rwX /app

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]