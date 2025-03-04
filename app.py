# flask app
# from llama_cpp import Llama
from flask import Flask

app = Flask(__name__)

# llm_sentence_embedding = Llama(
#     model_path = "/Users/katherineGoznikar/Desktop/legal_nlp/exclude_files/all-MiniLM-L6-v2.F16.gguf", 
#     embedding=True, 
#     n_ctx=1024, 
#     verbose=False)

@app.route('/')
def main():
    return "Hello World"

if __name__== '__main__':
    app.run(host='0.0.0.0:5002')