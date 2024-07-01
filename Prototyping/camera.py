import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2
import time
# Define the class_to_int and device just like in your training file
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
checkpoint = torch.load("checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model = model.to(device)

# Define your transforms - Please replace it based on how you transformed during training
transform = T.Compose([T.ToTensor()])

cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame from video
    ret, frame = cap.read()

    # Convert the frame to a PIL Image
    start_time = time.time()
    img = Image.fromarray(frame)

    # Perform the same preprocessing and prediction steps as before
    img_t = transform(img)
    img_t = img_t.unsqueeze(0)
    img_t = img_t.to(device)
    model.eval()
    threshold = 0.1
    with torch.no_grad():
        output = model(img_t)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    # Converting tensors to array, thresholding, and get labels
    boxes = output[0]['boxes'].data.cpu().numpy()
    scores = output[0]['scores'].data.cpu().numpy()
    labels = output[0]['labels'].data.cpu().numpy()
    boxes_th = boxes[scores >= threshold].astype(np.int32)
    scores_th = scores[scores >= threshold]
    labels_th = [int_to_class[labels[i]] for i in range(len(labels)) if scores[i] >= threshold]

    # Draw boxes and labels directly on the original captured frame
    for i in range(len(boxes_th)):
        print("thing found")
        box = boxes_th[i]
        score = scores_th[i]
        label = labels_th[i]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(frame, f'{label}: {score:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()

# finally, close the window
cv2.destroyAllWindows()