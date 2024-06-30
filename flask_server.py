from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask_socketio import SocketIO
import base64
import os
import random
import string
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np

# Define the class_to_int and device just like in your training
class_to_int = {'background': 0, 'red': 1, 'green': 2, 'yellow': 3} 
int_to_class =  {0: 'background', 1: 'red', 2: 'green', 3: 'yellow'}
device = torch.device('cpu')

# Create the same model structure
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
num_classes = len(class_to_int)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr = 0.0001)


# Load checkpoint
checkpoint = torch.load("checkpoint.pth",map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Remember to to(device)
model = model.to(device)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ehehz'
socketio = SocketIO(app, cors_allowed_origins="*",max_content_length=1000000000,max_http_buffer_size=1000000000)
CORS(app)

print("server ready")
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('message', 'Connected to server')

@socketio.on('disconnect')
def handle_disconnect():    
    print('Client disconnected')

@socketio.on('message')
def handle_message(data):
    print(len(data))
    # emit('message', data, broadcast=True)


@socketio.on('frame')
def handle_frame(data): 
    """
    Handle the received frame data.

    Args:
        data (str): The base64 encoded image data.

    Returns:
        None
    """
    print('Received frame')
    # Save the base64 data as an image file
    print(data)
    # data = data.replace('data:image/jpeg;base64,', '')
    # #maybe save with a uuid, and pass it into another function
    # with open('received_image.jpg', 'wb') as f:
    #     f.write(base64.b64decode(data))
    # print('Image saved to received_image.jpg')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000)