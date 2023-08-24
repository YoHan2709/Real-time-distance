import cv2
import warnings
import os
warnings.filterwarnings("ignore")
import termcolor

KNOWN_DISTANCE = 66
PERSON_WIDTH = 44


FONTS = cv2.FONT_HERSHEY_TRIPLEX

def detect_object(object):
    classes, scores, boxes = model.detect(object,0.5,0.5)
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        cv2.rectangle(object, box,(0,0,255), 1)
        cv2.putText(object,"{}:{}".format(class_names[classid],format(score,'.2f')), (box[0], box[1]-14), FONTS,0.6,(0,255,0), 1)

        if classid ==0: #person
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        
    return data_list

def cal_distance(f,W,w):
    return (W * f) / w

def cal_focalLength(d, W, w):
    return (w * d) / W

class_names = []
with open("classes.txt", "r") as objects_file:
    class_names = [e_g.strip() for e_g in objects_file.readlines()]

yoloNet = cv2.dnn.readNet('yolov7-tiny.weights', 'yolov7-tiny.cfg')

model = cv2.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

person_image_path = os.path.join("src","person.png")

person_data = detect_object(cv2.imread(person_image_path))
person_width_in_rf = person_data[0][1]

focal_person = cal_focalLength(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

try:
    capture = cv2.VideoCapture(0)
    while True:
        _,frame = capture.read()

        data = detect_object(frame) 
        for d in data:
            print(d)
            if d[0] =='person':
                distance = cal_distance(focal_person, PERSON_WIDTH, d[1])
                x,y = d[2]
        
            cv2.putText(frame,f"Distance:{format(distance,'.2f')}cm", (x+5,y+13), FONTS, 0.45,(0,255,0), 1)
            
            print("Distance of {} is {} cm".format(d[0],distance))

        cv2.imshow('frame',frame)
        exit_key_press = cv2.waitKey(1)

        if exit_key_press == ord('q'):
            break

    capture.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except cv2.error:
    termcolor.cprint("Select the WebCam or Camera index properly, in my case it is 2","red")