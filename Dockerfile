# Use a lightweight base image with Python
FROM python:3.10-slim

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install OS-level dependencies for audio and Whisper/TTS
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    curl \
    libportaudio2 \
    libasound-dev \
    portaudio19-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your code
COPY . .

# Run your script
CMD ["python", "main.py"]
