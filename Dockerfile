# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory in the container
WORKDIR $HOME/app

# Copy the requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download the embedding model
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')"

# Copy the rest of the application code
COPY --chown=user . .

# Create directories for data and database
RUN mkdir -p $HOME/app/data $HOME/app/chroma_db

# Set environment variables
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 7860

# Run the application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "app:app"]
