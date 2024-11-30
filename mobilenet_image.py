import subprocess
import os
import atexit

script = '''
import torch
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify
from io import BytesIO

app = Flask(__name__)

model = models.mobilenet_v2(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output[0], dim=0)
        top_prob, top_class = torch.max(probs, 0)

    return top_class.item(), top_prob.item()

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        image_bytes = file.read()
        class_id, probability = predict(image_bytes)
        return jsonify({'class_id': class_id, 'probability': probability}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5010)
'''

script_file = "temp_deploy_model.py"
env_name = "yxs-AIAgentDemo-FeIN"

conda_path = subprocess.run("which conda", shell=True, capture_output=True, text=True).stdout.strip()
conda_sh_path = os.path.join(os.path.dirname(conda_path), '..', 'etc', 'profile.d', 'conda.sh')

# Register cleanup function to run on exit
def cleanup():
    if os.path.exists(script_file):
        os.remove(script_file)
atexit.register(cleanup)

with open(script_file, "w") as file:
    file.write(script)
command = f"source {conda_sh_path} && conda activate {env_name} && python {script_file}"
print("MobileNet AIP address is: http://127.0.0.1:5010/predict")
subprocess.run(command, shell=True, executable="/bin/bash")
os.remove(script_file)
