FROM python:3.9

# Install Java (FIXED: use Java 17 instead of 11)
RUN apt-get update && apt-get install -y openjdk-17-jre && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip tools
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install -r requirements.txt

# Set Java environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Render uses dynamic port (IMPORTANT)
ENV PORT=5000

EXPOSE 5000

# Run Flask app
CMD ["python", "main.py"]