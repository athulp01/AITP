import keras
import cv2
import numpy as np
from mtcnn import MTCNN

class FaceNet:
    def __init__(self, path):
        self.model = keras.models.load_model(path)
        self.face_detector = MTCNN()

    def get_all_faces(self, image, threshold):
        result = []
        faces = self.face_detector.detect_faces(image)
        boxes = []
        for face in faces:
            if face["confidence"] >= threshold:
                box = face["box"]
                boxes.append(box)
                cropped = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                result.append(cropped)
        return boxes, result

    def get_embeddings(self, face_bgr):
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(face_rgb, (160, 160))
        resized = resized.astype(np.float32)
        mean, std = resized.mean(), resized.std()
        face = (resized - mean) / std
        expanded = np.expand_dims(face, 0)
        return self.model.predict(expanded)[0]