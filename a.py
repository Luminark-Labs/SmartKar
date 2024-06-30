import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
# Define the class_to_int and device just like in your training file
class_to_int = {'background': 0, 'red': 1, 'green': 2, 'yellow': 3} # it's not in your previous code snippet but you need to define it
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

# Then you can use model for prediction or further training.
input("model loaded, ender tonie")
# Load Image
img_path = '8281.png'
img = Image.open(img_path)

# Define your transforms - Please replace it based on how you transformed during training
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
    
    for x in range(len(labels)):
        if scores[x] > threshold:
            labels_th.append(int_to_class[labels[x]])
    #Appending results to csv
    for y in range(len(boxes_th)):
        
        #Bboxes, classname & image name
        x1 = boxes_th[y][0] * 4
        y1 = boxes_th[y][1] * 4
        x2 = boxes_th[y][2] * 4
        y2 = boxes_th[y][3] * 4
    import matplotlib.pyplot as plt

    # Draw the image
    plt.imshow(img.squeeze(0).permute(1, 2, 0))
    plt.axis('off')
    # Draw the boxes and labels
    for i in range(len(boxes_th)):
        box = boxes_th[i]
        score = scores_th[i]
        label = labels_th[i]
        
        # Draw the box
        print(box[0], box[1], box[2], box[3])
        plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor=label, linewidth=2))
        
        # Add the label
        plt.text(box[0], box[1]-10, f'{label}: {score:.2f}', color=label, fontsize=10)
    
    # Show the image with boxes and labels
    plt.show()
input("model loaded, ender tonie")
img_path = 'raw.jpeg'
img = Image.open(img_path)

# Define your transforms - Please replace it based on how you transformed during training
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
    
    for x in range(len(labels)):
        if scores[x] > threshold:
            labels_th.append(int_to_class[labels[x]])
    #Appending results to csv
    for y in range(len(boxes_th)):
        
        #Bboxes, classname & image name
        x1 = boxes_th[y][0] * 4
        y1 = boxes_th[y][1] * 4
        x2 = boxes_th[y][2] * 4
        y2 = boxes_th[y][3] * 4
    import matplotlib.pyplot as plt

    # Draw the image
    plt.imshow(img.squeeze(0).permute(1, 2, 0))
    plt.axis('off')
    # Draw the boxes and labels
    for i in range(len(boxes_th)):
        box = boxes_th[i]
        score = scores_th[i]
        label = labels_th[i]
        
        # Draw the box
        print(box[0], box[1], box[2], box[3])
        plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor=label, linewidth=2))
        
        # Add the label
        plt.text(box[0], box[1]-10, f'{label}: {score:.2f}', color=label, fontsize=10)
    
    # Show the image with boxes and labels
    plt.show()