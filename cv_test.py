import ObjectDetection_Yolov3 as ODY3
import cv2

img = cv2.imread("Sample Images/sample_001.png")
print(ODY3.find_objects(img))
print(ODY3.get_bounding_boxes(img))