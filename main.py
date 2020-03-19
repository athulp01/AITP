#%%
import cv2
from classifier import Classifier
import argparse

classifier = Classifier()

def predict_from_webcam():
    camera = cv2.VideoCapture(0)
    while True:
        return_value, image = camera.read()
        if return_value:
            cv2.imshow("camera" ,classifier.predict(image, 0.7))
    camera.release()

def train_classifier(path):
    classifier.train(path)

def predict_from_input_file(path):
    image = cv2.imread(path)
    cv2.imwrite("resilt.jpg", classifier.predict(image, 0.8))

if __name__ == "__main__":
    train_classifier()
