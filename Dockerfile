# Stage 1: Build

FROM python:3.9-slim as builder
# Install System Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    rm -rf /var/lib/lists

# Install Python Dependencies
COPY requirements.txt .
RUN pip install --user -r req_prod.txt

# Stage 2: Final
FROM python:3.9-slim

WORKDIR /app

# Copy only necessary files from the builder stage
COPY --from=builder /root/.local /root/.local
COPY . .

# Set Environment Variables
ENV PATH=/root/.local/bin:$PATH
ENV STREAMLIT_SERVER_PORT=8501

# Expose the port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]