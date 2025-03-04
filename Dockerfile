# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.13

# Install pip requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /usr/src/app
COPY . .
# COPY ./exclude_files/all-MiniLM-L6-v2.F16.gguf /app/llama_cpp
# COPY ./exclude_files/mixtral-8x7b-instruct-v0.1.Q2_K.gguf /app/llama_cpp

# do NOT use 5000
EXPOSE 4996

CMD ["python", "app.py"]