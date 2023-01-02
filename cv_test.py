import ObjectDetection_Yolov3 as ODY3
import cv2

img = cv2.imread("test.png")
print(ODY3.find_objects(img))