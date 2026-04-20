FROM python:3.9-slim

# Install Java using official Debian-compatible package set
RUN apt-get update && apt-get install -y default-jre-headless && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project
COPY . .

# Upgrade pip tools
RUN pip install --upgrade pip setuptools wheel

# Install dependencies
RUN pip install -r requirements.txt

# Set Java path (auto-detected path)
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="$JAVA_HOME/bin:$PATH"

# Render dynamic port
ENV PORT=5000

EXPOSE 5000

CMD ["python", "main.py"]