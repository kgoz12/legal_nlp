from llama_cpp import Llama
from flask import Flask, render_template, request, jsonify
from markupsafe import escape, Markup
import sys
import os

DATA_DIR = os.environ.get("DATA_DIR", "/data")

llm_q_and_a = Llama(
    os.path.join(DATA_DIR, "mixtral-8x7b-instruct-v0.1.Q2_K.gguf"), 
    n_ctx=512, 
    verbose=True)

app = Flask(__name__)

def predict(text_input):
    try:
        response = llm_q_and_a.create_chat_completion(
            seed=123,
            top_k = 0,
            temperature=0.05,
            max_tokens=512,
            messages = [
                {"role" : "user",
                "content": "I am a fuzzy bunny. I hip hop in the forest."},
                {"role": "assistant", 
                "content": text_input}
            ]
        )
        text_output = response['choices'][0]['message']['content']
    except:
        text_ouput = "something went wrong!"
    return text_output

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method =='POST':
        user_input=request.form['question']
        model_output = predict(user_input)
        return render_template('index.html', output=model_output, input_text=user_input)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=4996) # do NOT use 5000