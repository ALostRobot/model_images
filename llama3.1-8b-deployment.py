from typing import List, Optional
from flask import Flask, request, jsonify
import sys
sys.path.append('/usr1/project/yuanxiaosong/llama3/llama3')
from llama import Llama
import os


# Necessary environment variables for GPU inference
os.environ["RANK"] = "0"
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
# Flask APP
app = Flask(__name__)
# Global variable to hold the Llama model instance
generator = None


# Initialize model
def load_model(ckpt_dir: str, tokenizer_path: str, max_seq_len: int = 8192, max_batch_size: int = 4):
    global generator
    if generator is None:
        generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size
        )
    return generator


# Function to process a dialog and return the response
def generate_response(prompt: List[List[dict]], max_gen_len: Optional[int] = None, temperature: float = 0.6,
                      top_p: float = 0.9):
    results = generator.chat_completion(
        prompt,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    responses = []
    for p, result in zip(prompt, results):
        response = result['generation']['content']
        responses.append(response)

    return responses


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Invalid input, 'prompt' field is required."}), 400

    prompt = data['prompt']
    max_gen_len = data.get('max_gen_len', None)
    temperature = data.get('temperature', 0.6)
    top_p = data.get('top_p', 0.9)

    responses = generate_response(prompt, max_gen_len, temperature, top_p)

    return jsonify(responses)


if __name__ == "__main__":
    # Path to the model checkpoint and tokenizer
    ckpt_dir = "/usr1/project/yuanxiaosong/llama3/Llama-3.1-8B-Instruct/original/"
    tokenizer_path = "/usr1/project/yuanxiaosong/llama3/Llama-3.1-8B-Instruct/original/tokenizer.model"

    # Load the model before starting the API service
    load_model(ckpt_dir, tokenizer_path)

    print("Llama3.1-8B inference service started on 127.0.0.1: 5010/chat")

    # Run the Flask app
    app.run(host='0.0.0.0', port=5010)
