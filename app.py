import cv2
import numpy as np
import face_recognition
from flask import Flask, render_template, Response, redirect, url_for
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
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def attendance():
    attendance_list = []
    with open('Attendance.csv', 'r') as f:
        reader = csv.reader(f)
        attendance_list = list(reader)
    return render_template('attendance.html', attendance_list=attendance_list)

if __name__ == '__main__':
    app.run(debug=True)
