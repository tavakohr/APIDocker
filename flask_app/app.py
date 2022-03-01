from flask import Flask
from flask import request, jsonify
from PIL import Image
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import datetime
import os
import pickle
######################################################################

# Load Yolo
net = cv2.dnn.readNet('./weights/yolov3.weights', './cfg/yolov3.cfg')
classes = []
with open( "./coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN

imgpath='./walk.jpg'
def object_counter(imgpath):
    frame = cv2.imread(imgpath)

    height, width, channels = frame.shape
    #print(height, width, channels)

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #print(class_ids)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    output_dict = {}
    totalpeople = 0
    totalbackpack = 0
    totalsuitcase = 0

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)
            print(label)
            if label == "person":
                totalpeople += 1
            if label == "suitcase":
                totalsuitcase += 1
            if label == "backpack" or label == "handbag":
                totalbackpack += 1
    print(totalpeople,totalsuitcase,totalbackpack)
    x = datetime.datetime.now()
    previous_second = 0



    output_dict["datetime"] = x
    output_dict["totalpeople"] = totalpeople
    output_dict["totalsuitcase"] = totalsuitcase
    output_dict["totalbackpack"] = totalbackpack
    print(os.getcwd())
    print(output_dict)
    return(output_dict)



object_counter('./walk.jpg')
###############################################################################
server = Flask(__name__)

def run_request():
    index = int(request.json['index'])
    list = ['red', 'green', 'blue', 'yellow', 'black']
    return list[index]

@server.route('/', methods=['GET', 'POST'])
def index():
    output_dict = object_counter('./walk.jpg')
    response = {'text': 'The model is up and running. Send a POST request',   'output': output_dict}
    return jsonify(response)



@server.route('/post', methods=['GET', 'POST'])
def index_post():
    json_data = request.get_json()  # Get the POSTed json
    dict_data = json.loads(json_data)  # Convert json to dictionary

    img = dict_data["img"]  # Take out base64# str
    img = base64.b64decode(img)  # Convert image data converted to base64 to original binary data# bytes
    img = BytesIO(img)  # _io.Converted to be handled by BytesIO pillow
    img = Image.open(img)
    rgb_im = img.convert("RGB")

    # exporting the image
    rgb_im.save("tmp.jpg")

    output_dict = object_counter('tmp.jpg')
    response = {'text': 'the results is based on json POST request',
                'output': output_dict

                }

    return jsonify(response)

@server.route('/postimage', methods=['GET', 'POST'])
def index_postimage():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save("tmp.jpg")

    output_dict = object_counter("./tmp.jpg")
    response = {'text': 'The result is based on image file POST request',
                'output': output_dict
                }

    return jsonify(response)




if __name__ == "__main__":
    server.debug = True
    server.run(host="0.0.0.0", port=5000)


