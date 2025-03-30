from llama_cpp import Llama
import pyarrow as pa
import pandas as pd
import lance
from lance.vector import vec_to_table
from flask import Flask, render_template, request, jsonify
from markupsafe import escape, Markup
import sys
import os
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = os.environ.get("DATA_DIR", "/data")

llm_q_and_a=Llama(
    os.path.join(DATA_DIR, "mixtral-8x7b-instruct-v0.1.Q2_K.gguf"), 
    n_ctx=256, 
    verbose=False)

sentence_embeddings=Llama(
    model_path=os.path.join(DATA_DIR, "all-MiniLM-L6-v2.F16.gguf"),
    embedding=True,
    n_ctx=256,
    verbose=False)

ds = lance.dataset(uri=os.path.join(DATA_DIR, "chatbot.lance"))

app = Flask(__name__)

def predict(text_input):
    try:
        question_vector = sentence_embeddings.embed([text_input])
        K = 1
        response_data_frame = ds.to_table(
            nearest={"column": "vector",
                     "q": question_vector[0],
                     "metric": "dot",
                     "k": K}).to_pandas()
        for row in response_data_frame["text"]:
            response = llm_q_and_a.create_chat_completion(
                seed=123,
                top_k = 0,
                temperature=0.05,
                max_tokens=512,
                messages = [
                    {"role" : "user",
                    "content": row},
                    {"role": "assistant", 
                    "content": text_input}
                ]
            )
        text_output = response['choices'][0]['message']['content']
    except:
        text_ouput = "something went wrong!"
    return text_output

# GET loads the page
# POST sends user requests in via the submit button
@app.route('/', methods=['GET', 'POST'])
def home():
    # should I put something here that clears output?
    if request.method=='POST':
        user_input=request.form['question']
        model_output=predict(user_input)
        return render_template('index.html', output=model_output, input_text=user_input)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=4996) # do NOT use 5000


response_data_frame = ds.to_table(
    nearest={"column": "vector",
     "q": question_vector[0],
     "metric": "dot",
     "k": K}).to_pandas()
# text_output = response_data_frame["text"][0]
# response = llm_q_and_a.create_chat_completion(
#     seed=123,
#     top_k = 0,
#     temperature=0.05,
#     max_tokens=1024,
#     messages = [
#         {"role" : "user",
#         "content": response_data_frame["text"][0]},
#         {"role": "assistant", 
#         "content": text_input}
#     ]
# )
# text_output = response['choices'][0]['message']['content']