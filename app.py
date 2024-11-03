import base64
import os
import cv2
import numpy as np
import face_recognition
from flask import Flask, jsonify, render_template, Response, redirect, request, url_for
import csv
from datetime import datetime

app = Flask(__name__)

data = np.load('face_encodings.npz', allow_pickle=True)
encodeListKnown = data['encodings']
classNames = data['names']

cap = cv2.VideoCapture(0)

def markAttendance(name):
    with open('Attendance.csv', 'r+', newline='') as f:
        existing_data = f.readlines()
        nameList = [line.split(',')[0] for line in existing_data]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'\n{name},{dtString}')

def generate_frames():
    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                detected_name = "Unknown"
                if matches[matchIndex]:
                    detected_name = classNames[matchIndex]
                    markAttendance(detected_name)

                top, right, bottom, left = faceLoc
                top, right, bottom, left = top*4, right*4, bottom*4, left*4

                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(img, detected_name, (left, top - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/profile')
def pro():
    pass

@app.route('/attendance')
def attendance():
    attendance_list = []
    with open('Attendance.csv', 'r') as f:
        reader = csv.reader(f)
        attendance_list = list(reader)
    attendance_list = [i for i in attendance_list if i]  
    attendance_data = []

    for row in attendance_list:
        timestamp_str = row[1] 
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        if timestamp.hour == 8 and 0 <= timestamp.minute < 60:
            attendance_data.append(True)  
        else:
            attendance_data.append(False)  

    return render_template('attendance.html', attendance_list=attendance_list,attendance_data=attendance_data)

# Directory to save captured images
IMAGE_SAVE_PATH = 'ImagesAttendance'
if not os.path.exists(IMAGE_SAVE_PATH):
    os.makedirs(IMAGE_SAVE_PATH)

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/save_image', methods=['POST'])
def save_image():
    data = request.json
    name = data['name']
    img_data = data['image']

    # Decode the base64 image and save it
    img_data = base64.b64decode(img_data.split(',')[1])
    img_filename = os.path.join(IMAGE_SAVE_PATH, f"{name}.png")
    with open(img_filename, 'wb') as f:
        f.write(img_data)

    return jsonify({"status": "success", "message": f"Image saved as {img_filename}"})


if __name__ == '__main__':
    app.run(debug=True)
