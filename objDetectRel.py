# Ref doc: https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/README.md

#https://githubhelp.com/OlafenwaMoses/ImageAI
# To use ImageAI in your application developments, you must have installed the following dependencies before you install ImageAI :
#
# Python 3.7.6
# Tensorflow 2.4.0
# OpenCV
# Keras 2.4.3

#RetinaNet (Size = 145 mb, high performance and accuracy, with longer detection time)
#YOLOv3 (Size = 237 mb, moderate performance and accuracy, with a moderate detection time)
#TinyYOLOv3 (Size = 34 mb, optimized for speed and moderate performance, with fast detection time)

# 80 Obj supported
#person,  bicycle,  car, motorcycle, airplane, bus, train,  truck,  boat,  traffic light,  fire hydrant, stop_sign,
#parking meter,   bench,   bird,   cat,   dog,   horse,   sheep,   cow,   elephant,   bear,   zebra,
#giraffe,   backpack,   umbrella,   handbag,   tie,   suitcase,   frisbee,   skis,   snowboard,
#sports ball,   kite,   baseball bat,   baseball glove,   skateboard,   surfboard,   tennis racket,
#bottle,   wine glass,   cup,   fork,   knife,   spoon,   bowl,   banana,   apple,   sandwich,   orange,
#broccoli,   carrot,   hot dog,   pizza,   donot,   cake,   chair,   couch,   potted plant,   bed,
#dining table,   toilet,   tv,   laptop,   mouse,   remote,   keyboard,   cell phone,   microwave,   oven,
#toaster,   sink,   refrigerator,   book,   clock,   vase,   scissors,   teddy bear,   hair dryer,   toothbrush.


from imageai.Detection import ObjectDetection
import os
import cv2

execution_path = os.getcwd()
model_path='/Users/robertlin/.imageai'

detector = ObjectDetection()
#detector.setModelTypeAsYOLOv3()
#detector.setModelPath( os.path.join(model_path , "yolo.h5"))
#Model Option: RetinaNet
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(model_path , "resnet50_coco_best_v2.1.0.h5"))

detector.loadModel()
# detections, objsPath = detector.detectObjectsFromImage(\
#         input_image=os.path.join(execution_path , "data/objdetRetina/imgdayflowcustom.jpg"), \
#         output_image_path=os.path.join(execution_path , "data/objdetRetina/imgdayflowcustomNew.jpg"), \
#         minimum_percentage_probability=30, extract_detected_objects=True)

PROB_TH=46
OUT_FILE="data/objdetRetinaPort/imgRestCustom1New.jpg"
custom_objects = detector.CustomObjects(person=True, handbag=True)
detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,\
  input_image=os.path.join(execution_path , "data/objdetRetinaPort/imgRestCustom1.jpg"),\
  output_image_path=os.path.join(execution_path , OUT_FILE),\
  minimum_percentage_probability=PROB_TH, extract_detected_objects=True)

person=0
bag=0
# detections object is a tuple of list of dictionary when extract_detected_objects is True
for item in detections[0]:

    if item['name']=='person':
        person += 1
    elif item['name']=='handbag':
        bag += 1

# when extract_detected_objects=False
# for eachObject in detections:
#
#     if eachObject['name']=='person':
#         person += 1
#     elif eachObject['name']=='handbag':
#         bag += 1

titleObj = 'people count = '+str(person)+' bag count = '+str(bag)+' RetinaNet Det Th= '+str(PROB_TH)
image = cv2.imread(OUT_FILE)
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
cv2.putText(image, titleObj, (50, 30), cv2. FONT_HERSHEY_DUPLEX,
   0.7, ( 0, 255, 255), 1, cv2. LINE_AA)
#cv2.line(影像, 開始座標, 結束座標, 顏色, 線條寬度)
cv2.line(image, (50, 50), (320, 50), (0, 255, 255), 5)

cv2.imshow(OUT_FILE,image)

while (True):
    #wait for q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# for eachObject, eachObjPath in zip(detections, objsPath):
#     print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
#     print("Object's image saved in " + eachObjPath)
#     print("--------------------------------")
