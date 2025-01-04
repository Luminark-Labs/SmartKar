from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from flask_socketio import SocketIO, emit
from flask import Flask, render_template
import torchvision.transforms as T
from flask_cors import CORS
from PIL import Image
import torchvision
import numpy as np
import string
import random
import torch
import uuid
import time
import os
import json
from flask import jsonify

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

lights = {
}

def check_light(lightid):
    if lights[lightid]['color'] == 'red':
        return True

def GenerateLightID():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10))

def play_chime():
    emit('chime', 'true')


def GetCenters(boxes):
    centers = []
    for box in boxes:
        #[1606, 1243, 1685, 1280], [1568, 937, 1670, 975], [2067, 359, 2121, 504], [1613, 937, 1678, 978], [2138, 906, 2186, 992]
        x1, y1, x2, y2 = box
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        centers.append(center)
    return centers

def IsSameCenters(centers1, centers2):
    buffer = 100
    for center1 in centers1:
        for center2 in centers2:
            if abs(center1[0] - center2[0]) < buffer and abs(center1[1] - center2[1]) < buffer:
                print("save center!")
                return True
    return False

def ProcessLightInfo(centers, colors):
    timestamp = time.time()
    
    if not lights:  # If lights dictionary is empty
        for center, color in zip(centers, colors):
            light_id = GenerateLightID()
            lights[light_id] = {
                'position': center,
                'color': color,
                'last_seen': timestamp,
                'history': [(center, color)],
                'previousColor': color,
            }
    else:
        for center, color in zip(centers, colors):
            light_found = False
            for light_id in list(lights.keys()):
                if IsSameCenters([lights[light_id]['position']], [center]):
                    
                    lights[light_id]['color'] = color
                    lights[light_id]['last_seen'] = timestamp
                    lights[light_id]['history'].append((center, color))
                    light_found = True
                    if check_light(light_id):
                        print("play chime")
                    if lights[light_id]['previousColor'] == 'red' and 'color' == 'green':
                        print("play chime")
                    lights[light_id]['previous_color'] = color

                    break
            
            if not light_found:
                light_id = GenerateLightID()
                lights[light_id] = {
                    'position': center,
                    'color': color,
                    'last_seen': timestamp,
                    'history': [(center, color)],
                    'previous_color': None
                }

    # Uncomment this part if you want to remove lights not seen for more than 5 seconds
    # for light_id in list(lights.keys()):
    #     if timestamp - lights[light_id]['last_seen'] > 5:
    #         del lights[light_id]


def RunImageThroughModel(img_path):
    # Load Image
    img = Image.open(img_path)

    # Transformer
    transform = T.Compose([T.ToTensor()])

    # Transform Image
    img = transform(img)

    # Add extra batch dimension
    img = img.unsqueeze(0)

    # Pass Image Tensor to model - Remember to put the model in eval mode and use the correct device!
    model.eval()
    threshold = 0.8
    img = img.to(device)
    with torch.no_grad():
        output = model(img)

        # Converting tensors to array
        boxes = output[0]['boxes'].data.cpu().numpy()
        scores = output[0]['scores'].data.cpu().numpy()
        labels = output[0]['labels'].data.cpu().numpy()

        # Thresholding
        boxes_th = boxes[scores >= threshold].astype(np.int32)
        scores_th = scores[scores >= threshold]

        # int_to_class - labels
        labels_th = []
        for label in labels[scores >= threshold]:
            labels_th.append(int_to_class[label])

        #Converting to regular lists
        boxes_th = boxes_th.tolist()
        #boxes_th is like [x1, y1, x2, y2]
        scores_th = scores_th.tolist()

        BoxCenters = GetCenters(boxes_th)
        
        ToSendInfo = {
            "boxes": boxes_th,
            "scores": scores_th,
            "labels": labels_th,
            "centers": BoxCenters,
        }
        return ToSendInfo

def ProcessImage(img_path):
    Data = RunImageThroughModel(img_path)
    ProcessLightInfo(Data['centers'], Data['labels'])
    return Data


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
    random_id = uuid.uuid4()
    with open(f'./temp/{random_id}.jpg', 'wb') as f:
        f.write(data)    
    gr = ProcessImage(f'./temp/{random_id}.jpg')
    print(gr)
    
    emit('boxdrawing', gr   )
    os.remove(f'./temp/{random_id}.jpg')
    
    




@socketio.on('frame')
def handle_frame(data): 
    """
    Handle the received frame data.

    Args:
        data (str): The base64 encoded image data.

    Returns:
        None
    """
    #convert the binary data into an image, assign it with a random ID,
    #send to the ai, get the bounding boxes,

    #first of all, the boxes are not always going to be exactly the same, you need to find a way to identify each light, to make sure it doesnt get confused with other lights.
    #store the information of the light
    #check if chime needs to be played.




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