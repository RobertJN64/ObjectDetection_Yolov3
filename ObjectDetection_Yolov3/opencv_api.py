import numpy as np
import cv2
import os

def load_model():
    """
    Loads Yolo v3 model (called automtically on import)
    """
    yolov3_path = os.path.dirname(__file__) + '/Models/Yolo v3/'
    try:
        with open(yolov3_path + 'yolov3.weights'):
            pass
    except FileNotFoundError:
        print("Downloading model...")
        import wget
        import sys
        def bar_custom(current, total, _): #_ = width
            sys.stdout.write("\r")
            sys.stdout.write("Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total))

        wget.download('https://pjreddie.com/media/files/yolov3.weights', yolov3_path + 'yolov3.weights', bar=bar_custom)
        print()

    net: cv2.dnn_Net = cv2.dnn.readNet(yolov3_path + "yolov3.weights", yolov3_path + "yolov3.cfg")
    with open(yolov3_path + "yolov3.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    return net, classes, output_layers

def detect_objects(img, net: cv2.dnn_Net, outputLayers: list):
    """
    Returns DNN outputs for finding objects in image, uses cache if possible
    """
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    return net.forward(outputLayers)

def get_box_dimensions(outputs: list, thresh: float = 0.3):
    """
    Returns X, Y, width, height of objects
    """
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > thresh:
                center_x = detect[0] * 100
                center_y = detect[1] * 100
                w = round(detect[2] * 100, 3)
                h = round(detect[3] * 100, 3)
                x = round(center_x - w / 2, 3)
                y = round(center_y - h / 2, 3)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids