from llama_cpp import Llama

import json

llm = Llama(
    model_path="./model_files/qwen2.5-coder-0.5b-instruct-q2_k.gguf",
    n_gpu_layers=-1, # Uncomment to use GPU acceleration
    # seed=1337, # Uncomment to set a specific seed
    n_ctx=32_768 #n_ctx=32_768 # Uncomment to increase the context window
)

def save_chats_to_disk(chats_dict):
    with open("chats_dict.json", "w") as f:
        json.dump(chats_dict, f)

def load_chats_from_disk():
    try:
        with open("chats_dict.json", "r") as f:
            return json.load(f)
    except:
        return {}

chats_dict = load_chats_from_disk()
    
from flask import Flask, request, render_template
import markdown

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('homepage.html')

@app.route("/chat", methods=['POST', 'GET'])
def chat():
    message = request.args.get('message')
    chat_id = request.args.get('chat_id')

    chats_dict[chat_id] = chats_dict.get(chat_id, []) + [{ "role": "user", "content": message }]
    output = llm.create_chat_completion(
        messages=chats_dict[chat_id],
        repeat_penalty=1.2
    )

    chats_dict[chat_id] = chats_dict[chat_id] + [output["choices"][0]["message"]]

    save_chats_to_disk(chats_dict) # TODO: gets wrecked by multithreading

    return render_template('chat_message.html', message_user=message ,message_bot=markdown.markdown(output["choices"][0]["message"]["content"],extensions=['fenced_code']))

@app.route("/full_chat", methods=['POST','GET'])
def full_chat():
    chat_id = request.args.get('chat_id')
    return render_template('chat_conversation.html',chat_id=chat_id,conversation=chats_dict.get(chat_id, []),render_markdown=markdown.markdown)

@app.route("/chat_list", methods=['GET'])
def chat_list():
    return render_template('chat_list_small.html',chats=list(chats_dict.keys()))
