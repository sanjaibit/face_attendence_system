import os
import cv2
import numpy as np
import face_recognition

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    else:
        print(f"Warning: {cl} could not be loaded.")

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
        else:
            print("Warning: No face encoding found for an image.")
    return encodeList

encodeListKnown = findEncodings(images)
np.savez('face_encodings.npz', encodings=encodeListKnown, names=classNames)

print("Encodings saved successfully.")
