# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.13-slim

EXPOSE 5000

# # Keeps Python from generating .pyc files in the container
# ENV PYTHONDONTWRITEBYTECODE=1

# # Turns off buffering for easier container logging
# ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY app.py .
# COPY ./exclude_files/all-MiniLM-L6-v2.F16.gguf /app/llama_cpp
# COPY ./exclude_files/mixtral-8x7b-instruct-v0.1.Q2_K.gguf /app/llama_cpp

CMD ["python", "app.py"]