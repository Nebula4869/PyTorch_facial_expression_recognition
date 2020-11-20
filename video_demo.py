import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import dlib
import cv2


EMOTION_DICT = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise'}

onnx_session = onnxruntime.InferenceSession('models-2020-11-20-14-35/best-epoch71-0.7265.onnx')
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
plt.ion()
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        face_rects = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0)
        for face_rect in face_rects:
            cv2.rectangle(frame,
                          (face_rect.left(), face_rect.top()),
                          (face_rect.right(), face_rect.bottom()),
                          (255, 255, 255))
            face = frame[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right(), :]
            inputs = np.expand_dims(cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), (64, 64)), 0)
            inputs = np.expand_dims(inputs, 0).astype(np.float32) / 255.
            predictions = onnx_session.run(['output'], input_feed={'input': inputs})[0][0]
            predictions = (np.exp(predictions) / sum(np.exp(predictions)))
            emotion = EMOTION_DICT[int(np.argmax(predictions))]
            cv2.putText(frame, 'Emotion: {}'.format(emotion), (face_rect.left(), face_rect.top()), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
            # Draw Result
            plt.cla()
            plt.bar(x=np.arange(6), height=predictions * 100, width=0.4, align="center")
            plt.ylim(0, 100)
            plt.xticks(np.arange(6), ('anger', 'disgust', 'fear', 'happy', 'sad', 'surprise'))

        plt.show()
        cv2.imshow('', frame)
        cv2.waitKey(1)
