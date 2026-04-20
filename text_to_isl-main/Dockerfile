FROM python:3.9

# Install Java (required for Stanford Parser)
RUN apt-get update && apt-get install -y openjdk-11-jdk

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install dependencies
RUN pip install -r requirements.txt

# Set environment variables for Java
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Expose port (Render uses dynamic PORT)
EXPOSE 8080

# Run app
CMD ["python", "main.py"]