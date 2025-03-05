# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.13

# Install pip requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /usr/src/app
COPY app.py .

# QA model is too large to copy, need to mount it as external storage
# COPY ./exclude_files/mixtral-8x7b-instruct-v0.1.Q2_K.gguf .

# do NOT use 5000
EXPOSE 4996

CMD ["python", "app.py"]