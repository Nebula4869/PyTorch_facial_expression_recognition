import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import dlib
import cv2


EMOTION_DICT = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise'}

onnx_session = onnxruntime.InferenceSession('models-2020-11-20-14-35/best-epoch71-0.7265.onnx')
detector = dlib.get_frontal_face_detector()

img = cv2.imread('test.jpg')
face_rects = detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0)
plt.ion()
for face_rect in face_rects:
    cv2.rectangle(img,
                  (face_rect.left(), face_rect.top()),
                  (face_rect.right(), face_rect.bottom()),
                  (255, 255, 255))
    face = img[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right(), :]
    inputs = np.expand_dims(cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), (64, 64)), 0)
    inputs = np.expand_dims(inputs, 0).astype(np.float32) / 255.
    predictions = onnx_session.run(['output'], input_feed={'input': inputs})[0][0]
    predictions = (np.exp(predictions) / sum(np.exp(predictions)))
    emotion = EMOTION_DICT[int(np.argmax(predictions))]
    cv2.putText(img, 'Emotion: {}'.format(emotion), (face_rect.left(), face_rect.top()), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
    # Draw Result
    plt.cla()
    plt.bar(x=np.arange(6), height=predictions * 100, width=0.4, align="center")
    plt.ylim(0, 100)
    plt.xticks(np.arange(6), ('anger', 'disgust', 'fear', 'happy', 'sad', 'surprise'))

plt.show()
cv2.imshow('', img)
cv2.waitKey()
