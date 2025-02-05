# Use a lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

# Define working directory
WORKDIR $HOME/app

# Install essential system tools
RUN apt-get update -y && \
    apt-get install -y nano curl && \
    rm -rf /var/lib/apt/lists/*  # Reduce image size by clearing cache

# Copy project files
COPY . /home/user/app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy and make run.sh executable
COPY run.sh /run.sh
RUN chmod +x /run.sh

# Expose the correct port for Hugging Face
EXPOSE $PORT  

# Start the container with run.sh
CMD ["/run.sh"]