from llama_cpp import Llama
from flask import Flask, render_template, request

app = Flask(__name__)

# download the sentence embedding model file
llm_sentence_embedding  = Llama.from_pretrained(
	repo_id="leliuga/all-MiniLM-L6-v2-GGUF",
	filename="all-MiniLM-L6-v2.F16.gguf",
    embedding=True,
    verbose=False
)

# dowload the generative q&a model file
llm_q_and_a = Llama.from_pretrained(
    repo_id="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
    filename="mixtral-8x7b-instruct-v0.1.Q2_K.gguf", 
    n_threads=8,
    n_gpu_layers=30,
    verbose=False)

# @app.route('/')
# def hello():
#     return "Hello world"

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    
    question=request.form.get('question')
    print(question)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4996) # do NOT use 5000

# to run the Docker container from command line:
# cd ./Desktop/legal_nlp
# docker build -t flask-app .
# docker run -p 4996:4996 flask-app