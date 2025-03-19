# This is excellent guidance on using Docker for HuggingFace: https://www.docker.com/blog/llm-docker-for-local-and-hugging-face-hosting/
FROM python:3.13

# Install pip requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY app.py .
COPY templates/ ./templates/ 

ENV DATA_DIR=/data

# do NOT use 5000
EXPOSE 4996

CMD ["python", "app.py"]

# cd ./Desktop/legal_nlp
# docker build -t katiegoz312/my_flask_app .
# docker run -p 4996:4996 -v /Users/katherineGoznikar/Desktop/legal_nlp/exclude_files/models:/data --name=my_container katiegoz312/my_flask_app