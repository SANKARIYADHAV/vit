from flask import Flask, Response, render_template, request, jsonify
import cv2
import torch
import speech_recognition as sr
import threading

app = Flask(__name__)

# Load YOLO model once
print("Loading YOLO model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
print("Model Loaded Successfully.")

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Store recognized text globally with threading lock
recognized_text = {"user_speech": "", "receiver_speech": "", "user_sign": "", "receiver_sign": ""}
text_lock = threading.Lock()

# Video mapping for recognized words
video_mapping = {
    "hello": "static/videos/hello.mp4",
    "thank you": "static/videos/thankyou.mp4", 
    "i love you": "static/videos/iloveyou.mp4",
    "yes": "static/videos/yes.mp4",
    "no": "static/videos/no.mp4"
}

# Video capture objects
cap_user = cv2.VideoCapture(0)
cap_receiver = cv2.VideoCapture(1)

# Validate camera access
if not cap_user.isOpened():
    print("Error: User webcam not found!")
if not cap_receiver.isOpened():
    print("Warning: Receiver webcam not found! Switching to user webcam.")
    cap_receiver = cap_user


def detect_signs(cap, user_type):
    """ Detect hand signs and update recognized text. """
    global recognized_text

    while True:
        success, frame = cap.read()
        if not success:
            print(f"Camera error: {user_type}")
            break

        # YOLO detection
        results = model(frame)
        frame = results.render()[0]

        # Extract detected text
        detected_objects = results.pandas().xyxy[0]['name'].tolist()
        detected_text = " ".join(detected_objects) if detected_objects else "No Sign Detected"

        # Thread-safe update of recognized text
        with text_lock:
            recognized_text[user_type] = detected_text

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed_user')
def video_feed_user():
    """ Stream User's video feed with hand recognition. """
    return Response(detect_signs(cap_user, "user_sign"), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_receiver')
def video_feed_receiver():
    """ Stream Receiver's video feed with hand recognition. """
    return Response(detect_signs(cap_receiver, "receiver_sign"), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/recognize_speech', methods=['POST'])
def recognize_speech():
    """ Convert speech to text and play corresponding video. """
    global recognized_text
    data = request.json
    user_type = data.get("user_type")

    with sr.Microphone() as source:
        print(f"Listening for {user_type}...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio).lower()
        
        # Thread-safe update
        with text_lock:
            recognized_text[user_type] = text

        # Check if recognized word has a corresponding video
        video_url = video_mapping.get(text, "")

        return jsonify({"recognized_text": text, "video_url": video_url})
    except sr.UnknownValueError:
        return jsonify({"recognized_text": "Could not understand.", "video_url": ""})
    except sr.RequestError:
        return jsonify({"recognized_text": "API error.", "video_url": ""})


@app.route('/get_texts')
def get_texts():
    """ Provide recognized text for both user and receiver. """
    with text_lock:
        return jsonify(recognized_text)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
