from django.shortcuts import render, redirect
import cv2
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from time import time
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle

import tkinter as tk

from settings import BASE_DIR
# Create your views here.
def index(request):
    return render(request, 'index.html')
def errorImg(request):
    return render(request, 'error.html')


def gui_input(prompt):
    root = tk.Tk()
    # this will contain the entered string, and will
    # still exist after the window is destroyed
    var = tk.StringVar()

    # create the GUI
    label = tk.Label(root, text=prompt)
    entry = tk.Entry(root, textvariable=var)
    label.pack(side="left", padx=(20, 0), pady=20)
    entry.pack(side="right", fill="x", padx=(0, 20), pady=20, expand=True)

    # Let the user press the return key to destroy the gui
    entry.bind("<Return>", lambda event: root.destroy())

    # this will block until the window is destroyed
    root.mainloop()

    # after the window has been destroyed, we can't access
    # the entry widget, but we _can_ access the associated
    # variable
    value = var.get()
    return value


def trainer(request):
    import os
    from PIL import Image
    import numpy as np
    import cv2
    import pickle

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "../images")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    current_id = 0
    label_ids = {}

    x_train = []
    y_labels = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if (file.endswith("png") or file.endswith("jpg")):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path).replace(" ", "-"))
                print(label, path)
                if label in label_ids:
                    pass
                else:
                    label_ids[label] = current_id
                    current_id = current_id + 1
                    print(current_id)
                id_ = label_ids[label]
                # print(label_ids)
                # y_label.append(label) # some number
                # x_train.append(path) # verify  this image and turn into numpy array
                pil_image = Image.open(path).convert("L")  # grey scale conversion
                image_array = np.array(pil_image, "uint8")
                # print(image_array)
                #faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
                faces = face_cascade.detectMultiScale(image_array)
                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_labels.append(id_)

    with open("label.pickle", 'wb') as f:
        pickle.dump(label_ids, f)
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainer.yml")
    cv2.destroyAllWindows()
    return redirect('/')

def detect(request):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")
    user_id = "Unknown"
    labels = {"person_name": 1}
    with open("label.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}

    cap = cv2.VideoCapture(0)

    cap.set(3, 640)  # set Width
    cap.set(4, 480)  # set Height
    while (True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            print(x, y, w, h)
            roi_gray = gray[y:y + h, x:x + w]  # y coordinate start, y coordinate end
            roi_color = frame[y:y + h, x:x + w]
            # recognize Deep learned model predict keras tensorflow pytorch
            id_, conf = recognizer.predict(roi_gray)
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            stroke = 2
            if conf >= 45 and conf <= 95:
                print(id_)
                print(labels[id_])
                print("System is ", conf," percent confident that it is ", labels[id_])
                name = labels[id_]
                user_id = name
                cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
            else:
                cv2.putText(frame, user_id, (x, y), font, 1, color, stroke, cv2.LINE_AA)
            img_item = "my-image.png"
            cv2.imwrite(img_item, roi_gray)

            color = (255, 0, 0)  # BGR
            stroke = 2
            width = x + w
            height = y + h
            cv2.rectangle(frame, (x, y), (width, height), color, stroke)
        cv2.imshow('frame', frame)
        # cv2.imshow('gray', gray)

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press 'ESC' to quit
            break
        elif (user_id != "Unknown"):
            cv2.waitKey(1000)
            cap.release()
            cv2.destroyAllWindows()
            id = '/records/details/'+name
            print("Before routing: " + id)
            return redirect(id)
    cap.release()
    cv2.destroyAllWindows()
    return redirect('/')




face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    gray = cv2.cvtColor(img,cv2.COLOR_RGBA2RGB)
    faces = face_classifier.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face

def capture(request):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load functions
    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    count = 0
    new_dir ="test"
    #new_dir = gui_input("Get Folder Name")
    parent_dir = "/Users/Syed/UNT/SPRING2021/5214/vandana/Video-Auth-P2/images/"
    new_path = os.path.join(parent_dir, new_dir)
    os.mkdir(new_path)

    # Collect 100 samples of your face from webcam input
    while True:

        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (400, 400))
            face = cv2.cvtColor(face_extractor(frame), cv2.COLOR_RGBA2RGB )

            # Save file in specified directory with unique name
            file_name = 'train' + str(count) + '.jpg'

            # cv2.imwrite(file_name, face)

            cv2.imwrite(os.path.join(new_path ,file_name), face)
            # Put count on images and display live count
            cv2.putText(face, str(count), (200, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Cropper', face)

        else:
            print("Face not found")
            pass

        if count == 10 : #13 is the Enter Key
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting Samples Complete")
    return render(request, 'index.html')
