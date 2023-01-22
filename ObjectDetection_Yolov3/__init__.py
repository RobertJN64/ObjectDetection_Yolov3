import ObjectDetection_Yolov3.opencv_api as cv_api
import cv2

net, classes, output_layers = cv_api.load_model()

def find_objects(image, thresh: float = 0.3):
    """
    Returns a list of VisionObjects which can be used to find object location and size
    :param image: Image object (from load_image)
    :param thresh: Threshold to identify object, default is 30% (0.3)
    """
    outputs = cv_api.detect_objects(image, net, output_layers)
    boxes, confs, class_ids = cv_api.get_box_dimensions(outputs, thresh=thresh)
    v_indexes = cv2.dnn.NMSBoxes(boxes, confs, thresh, thresh) #Filter by seperated boxes
    return [classes[class_id] for index, class_id in enumerate(class_ids) if index in v_indexes]

def get_bounding_boxes(image, thresh: float = 0.3):
    outputs = cv_api.detect_objects(image, net, output_layers)
    boxes, confs, class_ids = cv_api.get_box_dimensions(outputs, thresh=thresh)
    v_indexes = cv2.dnn.NMSBoxes(boxes, confs, thresh, thresh)  # Filter by seperated boxes
    return [(classes[class_id], box) for index, (class_id, box)
            in enumerate(zip(class_ids, boxes)) if index in v_indexes]