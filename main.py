import cv2
import math
import argparse
#face detector 
def selectFace(net, frame, conf_threshold = 0.7): #confidence threshold to pass
    frameOpencvDnn = frame.copy() #copying detected frame to dnn
    frameHeight = frameOpencvDnn.shape[0] #getting number of rows of frame
    frameWidth = frameOpencvDnn.shape[1] #getting number of cols of frame
    #extracting blob for dnn
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False) #getting 4d blob from frame
    net.setInput(blob) #passing blob to dnn
    detections = net.forward()
    #loop used to generate faceList
    faceList = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        #if the confidence threshold is crossed attributes are selected
        if confidence > conf_threshold:   #Puts face on model for whole loop, if conf > threshold , extracts face, rectangle and store
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            faceList.append([x1, y1, x2, y2])       
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceList  #match returned

#main code

#creating a command line argument
parser = argparse.ArgumentParser()
parser.add_argument('--image')
args = parser.parse_args()
#models for our cnn
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
#mean values of our models
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(3-7)', '(8-12)', '(14-19)', '(20-30)', '(31-41)', '(44-54)', '(60-100)']
genderList = ['Male', 'Female']
#train using inbuilt functions
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
#for using our laptop camera
video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20 
#if no face in frame
while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg, faceList = selectFace(faceNet, frame)
    if not faceList:
        print("No face detected")

    for faceBox in faceList:
        face = frame[max(0, faceBox[1]-padding): #extrcts face but like google
                   min(faceBox[3]+padding, frame.shape[0]-1), max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]
      #preprocessing step to get image with same properties as trained images
      #brings image to bare minimum
        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)   #face,scale factor, size etc  
              #getting gender by using dnn
        genderNet.setInput(blob)
        genderPreds = genderNet.forward() #blobs keep on passing
        gender = genderList[genderPreds[0].argmax()] #max values argument returned
        print(f'Gender: {gender}')
        #getting age by using dnn
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')
        #printing age and gender
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detected age and gender", resultImg)
