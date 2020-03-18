from sklearn.svm import SVC
import os
from facenet import FaceNet
import cv2
import numpy as np
import pickle

class Classifier():
    def __init__(self):
        self.facenet = FaceNet("./facenet_keras.h5")
        self.model = None
        self.classes = None

    def train(self, path):
        self.classes = {idx:path for idx,path in enumerate(os.listdir(path))}
        pickle.dump(self.classes, open("class.pkl", "wb"))
        x_train = []
        y_train = []
        for key,_class in self.classes.items():
            for file in os.listdir(os.path.join(path, _class)):
                image = cv2.imread(os.path.join(path, _class, file))
                _, face = self.facenet.get_all_faces(image, 0.6)
                x_train.append(self.facenet.get_embeddings(face[0]))
                y_train.append(key)
        if len(self.classes) is 1:
            for _ in range(3):
                x_train.append(np.random.rand(128))
                y_train.append(len(self.classes))
        self.model = SVC(kernel="linear", probability=True)
        self.model.fit(x_train, y_train)
        pickle.dump(self.model, open("model.pkl", "wb"))
    
    def predict(self, image, threshold):
#   Returns image with names annoted it
#   image -> np.array
#   threshold -> int 
        if self.model is None:
            self.model = pickle.load(open("model.pkl", "rb"))
        if self.classes is None:
            self.classes = pickle.load(open("class.pkl", "rb"))
        boxes, faces = self.facenet.get_all_faces(image, 0.8)
        x_test = []
        if len(faces) is 0:
            return image
        for face in faces:
            x_test.append(self.facenet.get_embeddings(face))
        if len(x_test) is 1:
            faces = np.ravel(faces)
        probs = self.model.predict_proba(x_test)
        max_prob = np.argmax(probs, axis=1)
        for i, p in enumerate(max_prob):
            name = ""
            if probs[i][p] > threshold:
                name = self.classes[p]
            else:
                name = "Not recognized"
            font = cv2.FONT_HERSHEY_SIMPLEX
            pos_x = boxes[i][0]
            pos_y = boxes[i][1]+ boxes[i][3]
            cv2.putText(image, name, (pos_x, pos_y), font, 3.5, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("image", image)
            if cv2.waitKey(0):
                cv2.destroyAllWindows()
        return image

def debug():         
    classifier = Classifier()
    image = cv2.imread("./test.jpg")
    cv2.imwrite("result.jpg", classifier.predict(image, 0.85))

