
# Social Distancing Detector 

Hi Guys This is Day 1 of 15 days of Deep learning 

This project aims to detect people who are closer than the threshold in Covid era using Yolo weights.


##  Getting Started


This project is done in GoogleColab, You can do it in any ide like Pycharm or even in Kaggle

In golab 

This section downloads all the yolo pre-trained weights, yolo config, and also the class name which is coco.names
```bash
import os

# Create a directory to store YOLO model files
os.makedirs("yolo-coco", exist_ok=True)

# Download YOLO model weights
!wget -q -O yolo-coco/yolov3.weights https://pjreddie.com/media/files/yolov3.weights

# Download YOLO model configuration
!wget -q -O yolo-coco/yolov3.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg

# Download COCO names file
!wget -q -O yolo-coco/coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

```

Also you need a pesdestrian.mp4 video file , its been put my in the repo.
## Lessons Learned
- cv2: This is the OpenCV library, which is used for computer vision tasks such as image processing, video capture, and object detection.
- numpy: A library used for numerical computations in Python, often used to handle arrays and matrices efficiently.
```bash
import cv2
import numpy as np
```

🔗**Load the Yolo model Model**

cv2.dnn.readNetFromDarknet is a function to Load all the weights and conf files from yolo-coco directory
```bash
net = cv2.dnn.readNetFromDarknet(
    os.path.join(Config.MODEL_PATH, "yolov3.cfg"),
    os.path.join(Config.MODEL_PATH, "yolov3.weights")
)

```

🔗**Load the Yolo model class names from coco-names**

LABELS: A list containing the names of the classes that YOLO can detect, such as "person", "bicycle", "car", etc.
```bash
labelsPath = os.path.join(Config.MODEL_PATH, "coco.names")
LABELS = open(labelsPath).read().strip().split("\n")
```

**Get Yolo output layer**

- net.getLayerNames(): Gets the names of all the layers in the YOLO model.
- net.getUnconnectedOutLayers(): Gets the indices of the output layers of the YOLO model.
- ln[i - 1] for i in net.getUnconnectedOutLayers(): Creates a list of the names of the output layers. The -1 is used because OpenCV's layer indexing is 1-based, while Python's list indexing is 0-based.
- ln: A list of the names of the output layers of the YOLO model.
```bash
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

```

**The function def detect_people** - The detect_people function processes an input frame to detect people using the YOLO model, applies non-maxima suppression to eliminate overlapping detections, and returns a list of detected people with their confidence scores, bounding boxes, and centroids

### Bounding Box Coordinates

A bounding box is defined by its top-left corner and its width and height. 

  (x, y)
  
    +-------------------------------+
    |                               |
    |                               |
    |                               |
    |                               |
    |          (centerX, centerY)   |
    |               +               |
    |                               |
    |                               |
    |                               |
    +-------------------------------+
                       (x + w, y + h)

```bash

if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])    # Top-left corner of the bounding box
        (w, h) = (boxes[i][2], boxes[i][3])    # Width and height of the bounding box
        r = (confidences[i], (x, y, x + w, y + h), centroids[i])
        results.append(r)

```

### Calculate Social Distancing Violations

```bash
violate = set()

if len(results) >= 2:
    centroids = np.array([r[2] for r in results])
    D = np.zeros((len(centroids), len(centroids)), dtype="float")

    for i in range(0, len(centroids)):
        for j in range(i + 1, len(centroids)):
            D[i, j] = euclidean_dist(centroids[i], centroids[j])

            if D[i, j] < Config.MIN_DISTANCE:
                violate.add(i)
                violate.add(j)
```
- violate: Set to keep track of the indices of people violating social distancing.
- centroids: Array of centroids of detected people.
- D: Distance matrix to store pairwise distances between centroids.
- Nested loops calculate the Euclidean distance between each pair of centroids. If the distance is less than the minimum safe distance (Config.MIN_DISTANCE), both indices are added to the violate set



