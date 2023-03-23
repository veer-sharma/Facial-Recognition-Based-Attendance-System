import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime, timedelta
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import base64
import time
from connect import conn

#### Defining Flask App
app = Flask(__name__)

namem = ""
roll = ""

#### Saving Date today in 2 different formats
def datetoday():
    return date.today().strftime("%m_%d_%y")


def datetoday2():
    return date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
#cap = cv2.VideoCapture(0)

#### If these directories don't exist, create them
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
date_today = datetoday()


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    results = conn.read(f"SELECT * FROM \"{date_today}\"")
    return results


#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = (datetime.utcnow()+timedelta(hours=5.5)).strftime("%H:%M:%S")

    exists = conn.read(f'SELECT EXISTS(SELECT * FROM \"{date_today}\" WHERE roll={userid})')
    if exists[0][0] == 0:
        try:
            conn.insert(f'INSERT INTO \"{date_today}\"(name, roll, time) VALUES (%s, %s, %s)', (username, userid, current_time))
            print('Added Data in Database')
        except Exception as e:
            print(e)



################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    #no_of_tables = cur.execute('SHOW TABLES')
    #print('Tables', no_of_tables)
    conn.create(f'CREATE TABLE IF NOT EXISTS \"{date_today}\"(name VARCHAR(20), roll INT, time TIME)')
    userDetails = extract_attendance()
    return render_template('home.html', l=len(userDetails), totalreg=totalreg(),
                           datetoday2=datetoday2(), userDetails=userDetails)

@app.route('/video')
def video():
    return render_template('video.html')


# This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET', 'POST'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2(),
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    image_data = request.json['image']
    # Decode Base64-encoded image data and convert to NumPy array
    img_bytes = base64.b64decode(image_data.split(',')[1])
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if extract_faces(frame) != ():
        (x, y, w, h) = extract_faces(frame)[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
        face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
        identified_person = identify_face(face.reshape(1, -1))[0]
        add_attendance(identified_person)

    userDetails = extract_attendance()
    return render_template('home.html', l=len(userDetails), totalreg=totalreg(),
                           datetoday2=datetoday2(), userDetails=userDetails)


#### This function will run when we add a new user
@app.route('/start_capture', methods=['GET', 'POST'])
def start_capture():
    # Start capturing logic goes here
    global namem, roll
    namem = request.form.get('newusername')
    roll = request.form.get('newuserid')
    return render_template('capture.html')

@app.route('/save', methods=['POST'])
def save():
    dataUrl = request.json['dataUrl']
    index = request.json['index']
    directory = os.path.join(app.static_folder, 'faces', str(f'{namem}_{roll}'))
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'{namem}_{index-1}.jpg'
    filepath = f'{directory}'
    # Decode Base64-encoded image data and convert to NumPy array
    image_data = dataUrl
    img_bytes = base64.b64decode(image_data.split(',')[1])
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    cv2.imwrite(f'{filepath}\{filename}', img)
    print('Training Model')
    train_model()
    return '', 204


#### Our main function which runs the Flask App
if __name__ == '__main__':
    #app.run(debug=True)
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
