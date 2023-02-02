import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import yaml
from flask_mysqldb import MySQL

#### Defining Flask App
app = Flask(__name__)

db = yaml.safe_load(open('static/db.yaml'))

app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

mysql = MySQL(app)


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
    cur = mysql.connection.cursor()
    l = cur.execute(f"SELECT * FROM {date_today}")
    userDetails = cur.fetchall()
    #print(userDetails)
    mysql.connection.commit()
    cur.close()
    return l, userDetails


#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    cur = mysql.connection.cursor()
    cur.execute(f'SELECT EXISTS(SELECT * FROM {date_today} WHERE roll={userid})')
    axe = cur.fetchall()
    mysql.connection.commit()
    cur.close()
    if axe[0][0] == 0:
        try:
            cur = mysql.connection.cursor()
            cur.execute(f'INSERT INTO {date_today}(name, roll, time) VALUES (%s, %s, %s)',
                        (username, userid, current_time))
            mysql.connection.commit()
            cur.close()
            print('Added Data in Database')
        except Exception as e:
            print(e)


################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    cur = mysql.connection.cursor()
    cur.execute(f'CREATE TABLE IF NOT EXISTS {date_today}(name VARCHAR(20), roll INT, time TIME)')
    no_of_tables = cur.execute('SHOW TABLES')
    #print('Tables', no_of_tables)
    mysql.connection.commit()
    cur.close()
    l, userDetails = extract_attendance()
    return render_template('home.html', l=l, totalreg=totalreg(),
                           datetoday2=datetoday2(), userDetails=userDetails)


# This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2(),
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, frame = cap.read()
        if extract_faces(frame) != ():
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    l, userDetails = extract_attendance()
    return render_template('home.html', l=l, totalreg=totalreg(),
                           datetoday2=datetoday2(), userDetails=userDetails)


#### This function will run when we add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    noOfImg, j = 0, 0
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {noOfImg}/20', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername + '_' + str(noOfImg) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                noOfImg += 1
            j += 1
        if j == 200:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    l, userDetails = extract_attendance()
    return render_template('home.html', l=l, totalreg=totalreg(),
                           datetoday2=datetoday2(), userDetails=userDetails)


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=False)
