from flask import Flask, Response, render_template
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load your emotion detection model
emotion_model = load_model("model/emotion_model.h5")
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']  # As per training order

# Load age detection model
AGE_PROTOTXT = "model/age_deploy.prototxt"
AGE_MODEL = "model/age_net.caffemodel"
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTOTXT, AGE_MODEL)
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load gender detection model
GENDER_PROTOTXT = "model/gender_deploy.prototxt"
GENDER_MODEL = "model/gender_net.caffemodel"
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTOTXT, GENDER_MODEL)
GENDER_LABELS = ['Male', 'Female']

# Haarcascade face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_emotion_age_gender():
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Emotion Detection
            face_gray = gray[y:y+h, x:x+w]
            face_gray = cv2.resize(face_gray, (48, 48))
            face_gray = face_gray.astype("float32") / 255.0
            face_gray = img_to_array(face_gray)
            face_gray = np.expand_dims(face_gray, axis=0)  # Shape (1, 48, 48, 1)

            emotion_prediction = emotion_model.predict(face_gray, verbose=0)
            emotion_label = EMOTION_LABELS[np.argmax(emotion_prediction)]

            # Age Detection
            face_rgb = frame[y:y+h, x:x+w]
            age_blob = cv2.dnn.blobFromImage(face_rgb, 1.0, (227, 227),
                                             (78.426337, 87.768914, 114.895847), swapRB=False)
            age_net.setInput(age_blob)
            age_preds = age_net.forward()
            age_label = AGE_BUCKETS[np.argmax(age_preds)]

            # Gender Detection
            gender_blob = cv2.dnn.blobFromImage(face_rgb, 1.0, (227, 227),
                                                (78.426337, 87.768914, 114.895847), swapRB=False)
            gender_net.setInput(gender_blob)
            gender_preds = gender_net.forward()
            gender_label = GENDER_LABELS[np.argmax(gender_preds)]

            # Draw info on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 60), (x + w, y), (50, 50, 255), -1)

            cv2.putText(frame, f"{emotion_label}", (x + 5, y - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Age: {age_label}", (x + 5, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Gender: {gender_label}", (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Encode frame and yield as stream
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video_feed')
def video_feed():
    return Response(detect_emotion_age_gender(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
