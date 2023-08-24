import cv2
import os

FONTS = cv2.FONT_HERSHEY_TRIPLEX

def detect_object(image):
    classes, scores, boxes = model.detect(image, 0.5, 0.5)
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        if classid == 0:  # person
            data_list.append([class_names[classid], box])
    return data_list

class_names = []
with open("classes.txt", "r") as objects_file:
    class_names = [e_g.strip() for e_g in objects_file.readlines()]

yoloNet = cv2.dnn.readNet('yolov7-tiny.weights', 'yolov7-tiny.cfg')

model = cv2.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

person_image_path = os.path.join("src", "person.jpg")

image = cv2.imread(person_image_path)
person_data = detect_object(image)
print(person_data)

for (class_name, box) in person_data:
    box = [int(val) for val in box]
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, class_name, (x, y - 10), FONTS, 0.5, (0, 255, 0), 1)

cv2.imshow("Detected Persons", image)
cv2.waitKey(0)
cv2.destroyAllWindows()