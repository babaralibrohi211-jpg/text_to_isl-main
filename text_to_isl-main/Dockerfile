FROM python:3.9-slim

# Install Java (needed for Stanford Parser)
RUN apt-get update && apt-get install -y \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java

WORKDIR /root

COPY requirements.txt .
COPY . .

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Download spaCy small English model (~12MB, much lighter than stanza)
RUN python -m spacy download en_core_web_sm

EXPOSE 5000

CMD ["python", "main.py"]